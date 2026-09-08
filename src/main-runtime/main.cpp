// deepx-cpu-compute —— CPU tensor 计算 main-runtime。
// 嵌入 kvlang/runtime（C 核心）：自身只解释 myrwircaps = deepx/miaobyte·* 算子表（就地
// 零拷贝跑 deepx-core kernel），控制流/native/用户 rwfunc 写回 pc 交回 C 核心续跑；别的 runtime
// 的 rwir 才 handoff（跨 runtime 协作，暂无）。驱动循环对齐 kvlang/runtime-rs 的 drive_vid。
// 执行只查进程内 myrwircaps；/lib/<opcode> 注册仅供分布式调度，本进程派发不依赖它。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" { // kvlang 头为纯 C，无 extern "C" 守卫，C++ 下须显式包裹以取 C 链接名
#include <kvlang/kvlang_runtime.h>
#include <kvlang/kvlang_rwirext.h>
}
#include <kvspace/kvspace.h>

#include "hwy_compat.hpp" // 须先于 deepx-core SIMD kernel（补 highway 1.0.7 缺失的 IsAligned）

#include "deepx/tensor.hpp"
#include "deepx/shape_matmul.hpp"
#include "deepx/shape_reduce.hpp"
#include "deepx/shape_changeshape.hpp"
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"
#include "deepx/tensorfunc/matmul_miaobyte.hpp"
#include "deepx/tensorfunc/matmul_cblas.hpp"
#include "deepx/tensorfunc/reduce_miaobyte.hpp"
#include "deepx/tensorfunc/changeshape_miaobyte.hpp"
#include "deepx/tensorfunc/init_miaobyte.hpp"

#include "tensor_bridge.hpp"

#include <type_traits>

extern "C" int kvlangLayoutFile(const char *file, const char *dsn, char *entry_out,
                                uint32_t entry_len, char *err_out, uint32_t err_len);

using namespace deepx::rt;
namespace tf = deepx::tensorfunc;

namespace {

std::string take(char *p) {
    if (!p)
        return {};
    std::string s(p);
    free(p);
    return s;
}

std::string dsn() {
    const char *e = getenv("KVSPACE");
    return e && *e ? e : "redis://127.0.0.1:6379";
}

// 写 char/utf32 字符串（对齐 runtime-rs set_kv/set_tlv）：TlvEncode 取正典 head，
// WriteInPlace 就地失败则 WriteNewPlace 新箱，再拷 body。pc 写回即走此路。
void set_str(void *kv, const std::string &key, const std::string &val) {
    std::vector<uint8_t> raw;
    raw.reserve(val.size() * 4);
    for (unsigned char ch : val) {
        uint32_t u = ch; // pc 为 ASCII 路径，逐字节即码点
        raw.push_back(u & 0xff);
        raw.push_back((u >> 8) & 0xff);
        raw.push_back((u >> 16) & 0xff);
        raw.push_back((u >> 24) & 0xff);
    }
    int32_t dims[1] = {(int32_t)val.size()};
    uint8_t *tlv = nullptr;
    uint32_t tlen = 0;
    if (kvspaceTlvEncode("char/utf32", raw.data(), (uint32_t)raw.size(), dims, 1, &tlv, &tlen) != 0)
        return;
    kvspaceHead_t h{};
    if (kvspaceDecodeHead(tlv, tlen, &h) == 0) {
        uint8_t *bp = nullptr;
        char err[256] = {0};
        uint32_t bl = h.body_len > 0 ? (uint32_t)h.body_len : 0;
        if (kvspaceWriteInPlace(kv, key.c_str(), 1, bl, &bp, err, sizeof(err)) != 0)
            kvspaceWriteNewPlace(kv, key.c_str(), h.ref, h.storetype, h.ro, h.vid,
                                 (const char *)h.langtype, bl, &bp, err, sizeof(err));
        if (bp && bl)
            memcpy(bp, tlv + h.body_offset, bl);
    }
    free(tlv);
}

// ── myrwircaps：本 runtime 兑现的 rwir 表「key: opcode 串, id: handler 编号」──
// 一个 tensor 运算可有多套实现（miaobyte/cblas…），故按 /lib/deepx/<作者>·<op> 命名空间区分；
// 执行时以 opcode 查表得 (id, form)，按 form 取参、按 id 派发到对应 deepx-core kernel。
// form 决定实参形态：BINARY=A,B→C；SCALAR=A,标量→C；RSCALAR=标量,A→C（标量在前）；UNARY=A→C；
// CMP=A,B→bool mask；CMPS=A,标量→bool mask。
enum Form { F_BINARY, F_SCALAR, F_RSCALAR, F_UNARY, F_CMP, F_CMPS, F_MATMUL, F_REDUCE, F_RESHAPE, F_INIT };
enum OpId {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW, OP_MAX, OP_MIN,
    OP_ADDS, OP_SUBS, OP_MULS, OP_DIVS, OP_POWS, OP_MAXS, OP_MINS,
    OP_RSUBS, OP_RDIVS, OP_RPOWS,
    OP_SQRT, OP_LOG, OP_EXP, OP_SIN, OP_COS, OP_TAN, OP_NEG, OP_ABS,
    OP_EQ, OP_NE, OP_LT, OP_GT,
    OP_EQS, OP_NES, OP_LTS, OP_GTS,
    OP_MATMUL, OP_MATMUL_CBLAS,
    OP_SUM, OP_PROD, OP_RMAX, OP_RMIN,
    OP_RESHAPE, OP_TRANSPOSE,
    OP_CONSTANT, OP_ARANGE, OP_UNIFORM,
};
struct MyRwirCap {
    const char *op; // key
    int id;
    int form;
    int nr, nw;
    const char *sig;
};
const MyRwirCap myrwircaps[] = {
    {"deepx/miaobyte·add", OP_ADD, F_BINARY, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·sub", OP_SUB, F_BINARY, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·mul", OP_MUL, F_BINARY, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·div", OP_DIV, F_BINARY, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·pow", OP_POW, F_BINARY, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·max", OP_MAX, F_BINARY, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·min", OP_MIN, F_BINARY, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·addscalar", OP_ADDS, F_SCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·subscalar", OP_SUBS, F_SCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·mulscalar", OP_MULS, F_SCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·divscalar", OP_DIVS, F_SCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·powscalar", OP_POWS, F_SCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·maxscalar", OP_MAXS, F_SCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·minscalar", OP_MINS, F_SCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·rsubscalar", OP_RSUBS, F_RSCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·rdivscalar", OP_RDIVS, F_RSCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·rpowscalar", OP_RPOWS, F_RSCALAR, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·sqrt", OP_SQRT, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·log", OP_LOG, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·exp", OP_EXP, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·sin", OP_SIN, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·cos", OP_COS, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·tan", OP_TAN, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·neg", OP_NEG, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·abs", OP_ABS, F_UNARY, 1, 1, "any\nany"},
    {"deepx/miaobyte·equal", OP_EQ, F_CMP, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·notequal", OP_NE, F_CMP, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·less", OP_LT, F_CMP, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·greater", OP_GT, F_CMP, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·equalscalar", OP_EQS, F_CMPS, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·notequalscalar", OP_NES, F_CMPS, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·lessscalar", OP_LTS, F_CMPS, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·greaterscalar", OP_GTS, F_CMPS, 2, 1, "any\nany\nbool"},
    {"deepx/miaobyte·matmul", OP_MATMUL, F_MATMUL, 2, 1, "any\nany\nany"},
    {"deepx/cblas·matmul", OP_MATMUL_CBLAS, F_MATMUL, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·sum", OP_SUM, F_REDUCE, 3, 1, "any\nany\nany\nany"},
    {"deepx/miaobyte·prod", OP_PROD, F_REDUCE, 3, 1, "any\nany\nany\nany"},
    {"deepx/miaobyte·reducemax", OP_RMAX, F_REDUCE, 3, 1, "any\nany\nany\nany"},
    {"deepx/miaobyte·reducemin", OP_RMIN, F_REDUCE, 3, 1, "any\nany\nany\nany"},
    {"deepx/miaobyte·reshape", OP_RESHAPE, F_RESHAPE, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·transpose", OP_TRANSPOSE, F_RESHAPE, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·constant", OP_CONSTANT, F_INIT, 1, 1, "any\nany"},
    {"deepx/miaobyte·arange", OP_ARANGE, F_INIT, 2, 1, "any\nany\nany"},
    {"deepx/miaobyte·uniform", OP_UNIFORM, F_INIT, 3, 1, "any\nany\nany\nany"},
};

// 命中返 cap 指针，未命中返 nullptr。
const MyRwirCap *myrwircaps_find(const std::string &op) {
    for (const MyRwirCap &c : myrwircaps)
        if (op == c.op)
            return &c;
    return nullptr;
}

void register_myrwircaps(void *kv) {
    for (const MyRwirCap &c : myrwircaps)
        kvlang_rwirextRegister(kv, c.op, c.nr, c.nw, c.sig);
}

// 读参 idx 解析为标量（内联字面量或帧槽值）。
double read_scalar(void *kv, const std::string &pc, int idx) {
    std::string s = take(kvlang_rwirextResolveRead(kv, pc.c_str(), idx));
    return s.empty() ? 0.0 : strtod(s.c_str(), nullptr);
}

// 读参 idx 解析为 bool（true/1 为真）。
bool read_bool(void *kv, const std::string &pc, int idx) {
    std::string s = take(kvlang_rwirextResolveRead(kv, pc.c_str(), idx));
    return s == "true" || (!s.empty() && strtod(s.c_str(), nullptr) != 0.0);
}

// 读参 idx（整型数组，如 dims/new_shape/dim_order）借用整块 → vector<int>。
std::vector<int> read_int_vec(void *kv, const std::string &pc, int idx) {
    View v = read_view(kv, take(kvlang_rwirextResolveReadPath(kv, pc.c_str(), idx)));
    std::vector<int> out;
    if (!v.found)
        return out;
    std::string k = kind_of(v.kindexpr);
    int n = v.numel();
    if (k == "int64") {
        auto *p = (int64_t *)v.body();
        for (int i = 0; i < n; i++)
            out.push_back((int)p[i]);
    } else if (k == "int32") {
        auto *p = (int32_t *)v.body();
        for (int i = 0; i < n; i++)
            out.push_back((int)p[i]);
    }
    return out;
}

template <typename T>
void dump_out(const std::string &op, const deepx::Tensor<T> &C) {
    if (!getenv("DEEPX_DUMP"))
        return;
    fprintf(stderr, "[deepx-cpu] %s ->", op.c_str());
    for (int i = 0; i < C.shape.size; i++)
        fprintf(stderr, " %g", (double)C.data[i]);
    fprintf(stderr, "\n");
}

// A op B -> C（同形同型）。
template <typename T>
void do_binary(void *kv, int id, const std::string &op, const View &va, const View &vb,
               const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto B = borrow<T>(vb.body(), vb.dims());
    auto C = alloc_out<T>(kv, out, va.kindexpr, va.dims());
    switch (id) {
    case OP_ADD: tf::add<tf::miaobyte, T>(A, B, C); break;
    case OP_SUB: tf::sub<tf::miaobyte, T>(A, B, C); break;
    case OP_MUL: tf::mul<tf::miaobyte, T>(A, B, C); break;
    case OP_MAX: tf::max<tf::miaobyte, T>(A, B, C); break;
    case OP_MIN: tf::min<tf::miaobyte, T>(A, B, C); break;
    case OP_DIV:
        if constexpr (std::is_floating_point_v<T>) // 整型 SIMD 无除法，仅浮点
            tf::div<tf::miaobyte, T>(A, B, C);
        else
            fprintf(stderr, "deepx-cpu: %s 暂不支持整型\n", op.c_str());
        break;
    case OP_POW:
        if constexpr (std::is_floating_point_v<T>)
            tf::pow<tf::miaobyte, T>(A, B, C);
        else
            fprintf(stderr, "deepx-cpu: %s 暂不支持整型\n", op.c_str());
        break;
    }
    dump_out(op, C);
}

// A @ B -> C（矩阵乘；输出形状由 matmul_shape 推导，与输入不同）。
// use_cblas=true 走 cblas（仅 float32/float64），否则 miaobyte 通用实现。
template <typename T>
void do_matmul(void *kv, bool use_cblas, const std::string &op, const View &va, const View &vb,
               const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto B = borrow<T>(vb.body(), vb.dims());
    deepx::Shape cs = deepx::matmul_shape(A.shape, B.shape);
    auto C = alloc_out<T>(kv, out, make_kindexpr(cs.shape, kind_of(va.kindexpr)), cs.shape);
    if (use_cblas) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
            tf::matmul<tf::cblas, T>(A, B, C);
        else
            fprintf(stderr, "deepx-cpu: %s(cblas) 仅支持 float32/float64\n", op.c_str());
    } else {
        tf::matmul<tf::miaobyte, T>(A, B, C);
    }
    dump_out(op, C);
}

// reduce(A, dims, keepdims) -> B（输出形状由 reducedShape 推导）。
template <typename T>
void do_reduce(void *kv, int id, const std::string &op, const View &va,
               const std::vector<int> &dims, bool keepdims, const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    std::vector<int> od = deepx::reducedShape(va.dims(), dims, keepdims);
    auto B = alloc_out<T>(kv, out, make_kindexpr(od, kind_of(va.kindexpr)), od);
    switch (id) {
    case OP_SUM: tf::sum<tf::miaobyte, T>(A, dims, keepdims, B); break;
    case OP_PROD: tf::prod<tf::miaobyte, T>(A, dims, keepdims, B); break;
    case OP_RMAX: tf::reducemax<tf::miaobyte, T>(A, dims, keepdims, B); break;
    case OP_RMIN: tf::reducemin<tf::miaobyte, T>(A, dims, keepdims, B); break;
    }
    dump_out(op, B);
}

// reshape(A, new_shape) / transpose(A, dim_order) -> B。
template <typename T>
void do_reshape(void *kv, int id, const std::string &op, const View &va,
                const std::vector<int> &param, const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    std::vector<int> od = (id == OP_TRANSPOSE) ? deepx::transposeShape(va.dims(), param) : param;
    auto B = alloc_out<T>(kv, out, make_kindexpr(od, kind_of(va.kindexpr)), od);
    if (id == OP_TRANSPOSE)
        tf::transpose<tf::miaobyte, T>(A, param, B);
    else
        tf::reshape<tf::miaobyte, T>(A, param, B);
    dump_out(op, B);
}

// init：填充声明形状的输出张量（形状取自写槽已声明的 kindexpr）。
template <typename T>
void do_init(int id, const std::string &op, deepx::Tensor<T> &Tn, double p0, double p1, double p2) {
    switch (id) {
    case OP_CONSTANT: tf::constant<tf::miaobyte, T>(Tn, (T)p0); break;
    case OP_ARANGE: tf::arange<tf::miaobyte, T>(Tn, (T)p0, (T)p1); break;
    case OP_UNIFORM: tf::uniform<tf::miaobyte, T>(Tn, (T)p0, (T)p1, (unsigned int)p2); break;
    }
    dump_out(op, Tn);
}

// A op scalar -> C（同形同型）。
template <typename T>
void do_scalar(void *kv, int id, const std::string &op, const View &va, double sv,
               const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto C = alloc_out<T>(kv, out, va.kindexpr, va.dims());
    T v = (T)sv;
    switch (id) {
    case OP_ADDS: tf::addscalar<tf::miaobyte, T>(A, v, C); break;
    case OP_SUBS: tf::subscalar<tf::miaobyte, T>(A, v, C); break;
    case OP_MULS: tf::mulscalar<tf::miaobyte, T>(A, v, C); break;
    case OP_MAXS: tf::maxscalar<tf::miaobyte, T>(A, v, C); break;
    case OP_MINS: tf::minscalar<tf::miaobyte, T>(A, v, C); break;
    case OP_DIVS:
        if constexpr (std::is_floating_point_v<T>)
            tf::divscalar<tf::miaobyte, T>(A, v, C);
        else
            fprintf(stderr, "deepx-cpu: %s 暂不支持整型\n", op.c_str());
        break;
    case OP_POWS:
        if constexpr (std::is_floating_point_v<T>)
            tf::powscalar<tf::miaobyte, T>(A, v, C);
        else
            fprintf(stderr, "deepx-cpu: %s 暂不支持整型\n", op.c_str());
        break;
    }
    dump_out(op, C);
}

// scalar op A -> C（标量在前）。
template <typename T>
void do_rscalar(void *kv, int id, const std::string &op, double sv, const View &va,
                const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto C = alloc_out<T>(kv, out, va.kindexpr, va.dims());
    T v = (T)sv;
    switch (id) {
    case OP_RSUBS: tf::rsubscalar<tf::miaobyte, T>(v, A, C); break;
    case OP_RDIVS:
        if constexpr (std::is_floating_point_v<T>)
            tf::rdivscalar<tf::miaobyte, T>(v, A, C);
        else
            fprintf(stderr, "deepx-cpu: %s 暂不支持整型\n", op.c_str());
        break;
    case OP_RPOWS:
        if constexpr (std::is_floating_point_v<T>)
            tf::rpowscalar<tf::miaobyte, T>(v, A, C);
        else
            fprintf(stderr, "deepx-cpu: %s 暂不支持整型\n", op.c_str());
        break;
    }
    dump_out(op, C);
}

// op(A) -> C（一元；超越函数仅浮点，sqrt 全类型）。
template <typename T>
void do_unary(void *kv, int id, const std::string &op, const View &va, const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto C = alloc_out<T>(kv, out, va.kindexpr, va.dims());
    if (id == OP_SQRT) {
        tf::sqrt<tf::miaobyte, T>(A, C);
    } else if (id == OP_NEG) {
        tf::neg<tf::miaobyte, T>(A, C);
    } else if (id == OP_ABS) {
        tf::abs<tf::miaobyte, T>(A, C);
    } else if constexpr (std::is_floating_point_v<T>) {
        switch (id) {
        case OP_LOG: tf::log<tf::miaobyte, T>(A, C); break;
        case OP_EXP: tf::exp<tf::miaobyte, T>(A, C); break;
        case OP_SIN: tf::sin<tf::miaobyte, T>(A, C); break;
        case OP_COS: tf::cos<tf::miaobyte, T>(A, C); break;
        case OP_TAN: tf::tan<tf::miaobyte, T>(A, C); break;
        }
    } else {
        fprintf(stderr, "deepx-cpu: %s 仅支持浮点\n", op.c_str());
    }
    dump_out(op, C);
}

// A cmp B -> bool mask（equal/notequal 带 epsilon=1e-6）。
template <typename T>
void do_cmp(void *kv, int id, const std::string &op, const View &va, const View &vb,
            const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto B = borrow<T>(vb.body(), vb.dims());
    auto M = alloc_out<bool>(kv, out, with_kind(va.kindexpr, "bool"), va.dims());
    switch (id) {
    case OP_EQ: tf::equal<tf::miaobyte, T, bool>(A, B, 1e-6f, M); break;
    case OP_NE: tf::notequal<tf::miaobyte, T, bool>(A, B, 1e-6f, M); break;
    case OP_LT: tf::less<tf::miaobyte, T, bool>(A, B, M); break;
    case OP_GT: tf::greater<tf::miaobyte, T, bool>(A, B, M); break;
    }
    dump_out(op, M);
}

// A cmp scalar -> bool mask。
template <typename T>
void do_cmps(void *kv, int id, const std::string &op, const View &va, double sv,
             const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto M = alloc_out<bool>(kv, out, with_kind(va.kindexpr, "bool"), va.dims());
    T v = (T)sv;
    switch (id) {
    case OP_EQS: tf::equalscalar<tf::miaobyte, T, bool>(A, v, 1e-6f, M); break;
    case OP_NES: tf::notequalscalar<tf::miaobyte, T, bool>(A, v, 1e-6f, M); break;
    case OP_LTS: tf::lessscalar<tf::miaobyte, T, bool>(A, v, M); break;
    case OP_GTS: tf::greaterscalar<tf::miaobyte, T, bool>(A, v, M); break;
    }
    dump_out(op, M);
}

#define DISPATCH_T(k, CALL)                                                                      \
    do {                                                                                           \
        if ((k) == "float64") CALL(double);                                                        \
        else if ((k) == "float32") CALL(float);                                                    \
        else if ((k) == "int64") CALL(int64_t);                                                     \
        else if ((k) == "int32") CALL(int32_t);                                                     \
        else fprintf(stderr, "deepx-cpu: %s 不支持 dtype %s\n", cap.op, (k).c_str());              \
    } while (0)

void dispatch(void *kv, const MyRwirCap &cap, const std::string &pc) {
    std::string op = cap.op;
    std::string out = take(kvlang_rwirextResolveWrite(kv, pc.c_str(), 0));
    if (out.empty()) {
        fprintf(stderr, "deepx-cpu: %s 缺写参 @ %s\n", op.c_str(), pc.c_str());
        return;
    }
    if (cap.form == F_RSCALAR) { // 标量在前、张量在读参 1
        double sv = read_scalar(kv, pc, 0);
        View vb = read_view(kv, take(kvlang_rwirextResolveReadPath(kv, pc.c_str(), 1)));
        if (!vb.found) {
            fprintf(stderr, "deepx-cpu: %s 缺张量参 @ %s\n", op.c_str(), pc.c_str());
            return;
        }
        std::string k = kind_of(vb.kindexpr);
#define C(T) do_rscalar<T>(kv, cap.id, op, sv, vb, out)
        DISPATCH_T(k, C);
#undef C
        return;
    }
    if (cap.form == F_INIT) { // 输出张量形状取自写槽已声明的值（layout 阶段已布局），读参仅标量
        View vo = read_view(kv, out);
        if (!vo.found) {
            fprintf(stderr, "deepx-cpu: %s 写槽 %s 未布局，无法取形状\n", op.c_str(), out.c_str());
            return;
        }
        std::string k = kind_of(vo.kindexpr);
        double p0 = read_scalar(kv, pc, 0);
        double p1 = cap.nr > 1 ? read_scalar(kv, pc, 1) : 0.0;
        double p2 = cap.nr > 2 ? read_scalar(kv, pc, 2) : 0.0;
#define C(T)                                                                                       \
    do {                                                                                           \
        auto Tn = alloc_out<T>(kv, out, vo.kindexpr, vo.dims());                                    \
        do_init<T>(cap.id, op, Tn, p0, p1, p2);                                                     \
    } while (0)
        DISPATCH_T(k, C);
#undef C
        return;
    }
    // 其余形态：主张量在读参 0
    View va = read_view(kv, take(kvlang_rwirextResolveReadPath(kv, pc.c_str(), 0)));
    if (!va.found) {
        fprintf(stderr, "deepx-cpu: %s 缺张量参 @ %s\n", op.c_str(), pc.c_str());
        return;
    }
    std::string k = kind_of(va.kindexpr);
    if (cap.form == F_BINARY || cap.form == F_CMP || cap.form == F_MATMUL) {
        View vb = read_view(kv, take(kvlang_rwirextResolveReadPath(kv, pc.c_str(), 1)));
        if (!vb.found) {
            fprintf(stderr, "deepx-cpu: %s 缺张量参 @ %s\n", op.c_str(), pc.c_str());
            return;
        }
        if (cap.form == F_BINARY) {
#define C(T) do_binary<T>(kv, cap.id, op, va, vb, out)
            DISPATCH_T(k, C);
#undef C
        } else if (cap.form == F_CMP) {
#define C(T) do_cmp<T>(kv, cap.id, op, va, vb, out)
            DISPATCH_T(k, C);
#undef C
        } else { // F_MATMUL
#define C(T) do_matmul<T>(kv, cap.id == OP_MATMUL_CBLAS, op, va, vb, out)
            DISPATCH_T(k, C);
#undef C
        }
    } else if (cap.form == F_SCALAR || cap.form == F_CMPS) {
        double sv = read_scalar(kv, pc, 1);
        if (cap.form == F_SCALAR) {
#define C(T) do_scalar<T>(kv, cap.id, op, va, sv, out)
            DISPATCH_T(k, C);
#undef C
        } else {
#define C(T) do_cmps<T>(kv, cap.id, op, va, sv, out)
            DISPATCH_T(k, C);
#undef C
        }
    } else if (cap.form == F_REDUCE) {
        std::vector<int> dims = read_int_vec(kv, pc, 1);
        bool keepdims = read_bool(kv, pc, 2);
#define C(T) do_reduce<T>(kv, cap.id, op, va, dims, keepdims, out)
        DISPATCH_T(k, C);
#undef C
    } else if (cap.form == F_RESHAPE) {
        std::vector<int> param = read_int_vec(kv, pc, 1);
#define C(T) do_reshape<T>(kv, cap.id, op, va, param, out)
        DISPATCH_T(k, C);
#undef C
    } else { // F_UNARY
#define C(T) do_unary<T>(kv, cap.id, op, va, out)
        DISPATCH_T(k, C);
#undef C
    }
}

// 从 pc 续跑 vid 到结束：就地批处理 tensor.*，非本 caps 的扩展 rwir handoff，其余写回 pc。
void drive_vid(kvlangRuntime_t *rt, void *kv, const std::string &vid) {
    for (;;) {
        char *pc_out = nullptr;
        int rc = kvlangRuntimeExecuteVthread(rt, vid.c_str(), &pc_out);
        if (rc == 0)
            break;
        if (rc != 1) {
            fprintf(stderr, "deepx-cpu: vthread %s error rc=%d\n", vid.c_str(), rc);
            exit(1);
        }
        std::string c = take(pc_out), stop_op;
        for (;;) {
            std::string params = take(kvlang_rwirextParams(kv, c.c_str()));
            std::string op = params.substr(0, params.find('\n'));
            const MyRwirCap *cap = myrwircaps_find(op);
            if (!cap) {
                stop_op = op;
                break;
            }
            dispatch(kv, *cap, c);
            c = take(kvlang_rwirextNextPc(c.c_str()));
        }
        // pc 可能属子 vthread：目标 vid 由 pc 第 3 段导出。
        std::string sub = vid;
        {
            size_t a = c.find('/', 1);
            if (a != std::string::npos) {
                size_t b = c.find('/', a + 1);
                if (b != std::string::npos)
                    sub = c.substr(a + 1, b - a - 1);
            }
        }
        if (!stop_op.empty() && myrwircaps_find(stop_op)) {
            if (kvlang_rwirextHandoff(kv, sub.c_str(), c.c_str()) != 0) {
                fprintf(stderr, "deepx-cpu: handoff %s 失败 @ %s\n", stop_op.c_str(), c.c_str());
                exit(1);
            }
        } else {
            set_str(kv, "/vthread/" + sub + "/‥pc", c);
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.kv>|<entry>\n", argv[0]);
        return 1;
    }
    std::string arg = argv[1], d = dsn(), entry;

    // 须先 connect+register，再 layout：connect 会初始化空间，layout 早于它则 /lib 被清（对齐 runtime-rs 顺序）。
    kvlangRuntime_t *rt = kvlangRuntimeConnect(d.c_str());
    if (!rt) {
        fprintf(stderr, "deepx-cpu: kvlangRuntimeConnect 失败: %s\n", d.c_str());
        return 1;
    }
    void *kv = kvlangRuntimeKvspaceHandle(rt); // 复用同一句柄（durable 惰性 flush 只同句柄相干）
    // 标量算术/比较/控制流/=拷贝/用户函数是 runtime-c 执行核心固有能力（ExecuteVthread 内在），无需注册；
    // 本 runtime 只叠加自身 deepx/<author>·* tensor 算子表。
    register_myrwircaps(kv);

    // <file.kv>：layout 进 kvspace，入口取 layout 产物；否则 arg 即已入库入口。
    if (arg.size() > 3 && arg.compare(arg.size() - 3, 3, ".kv") == 0) {
        char eb[512] = {0}, err[512] = {0};
        if (kvlangLayoutFile(arg.c_str(), d.c_str(), eb, sizeof(eb), err, sizeof(err)) != 0) {
            fprintf(stderr, "deepx-cpu: layout 失败: %s\n", err);
            return 1;
        }
        entry = eb[0] ? eb : "test";
    } else {
        entry = arg;
    }

    std::string vid = take(kvlangRuntimeBootstrap(rt, entry.c_str(), nullptr, 0));
    if (vid.empty()) {
        fprintf(stderr, "deepx-cpu: bootstrap %s 失败\n", entry.c_str());
        return 1;
    }
    drive_vid(rt, kv, vid);
    return 0;
}
