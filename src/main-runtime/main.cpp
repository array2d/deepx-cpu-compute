// deepx-cpu-compute —— CPU tensor 计算 main-runtime。
// 嵌入 kvlang/runtime（C 核心）：自身只解释 myrwircaps = tensor.* 算子表（就地零拷贝跑
// deepx-core kernel），控制流/native/用户 rwfunc 交回 C 核心，非 tensor 扩展 rwir handoff。
// 驱动循环对齐 kvlang/runtime-rs 的 drive_vid。

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
#include "deepx/tensorfunc/elementwise_miaobyte.hpp"

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

// ── myrwircaps：CPU tensor 算子表（逐槽 kindexpr 签名）───────────────────
struct Cap {
    const char *op;
    int nr, nw;
    const char *sig;
};
const Cap CAPS[] = {
    {"tensor.add", 2, 1, "any\nany\nany"},
    {"tensor.sub", 2, 1, "any\nany\nany"},
    {"tensor.mul", 2, 1, "any\nany\nany"},
    {"tensor.div", 2, 1, "any\nany\nany"},
};

bool in_myrwircaps(const std::string &op) {
    for (const Cap &c : CAPS)
        if (op == c.op)
            return true;
    return false;
}

void register_caps(void *kv) {
    for (const Cap &c : CAPS)
        kvlang_rwirextRegister(kv, c.op, c.nr, c.nw, c.sig);
}

// A op B -> C（同形同型，逐 dtype 展开）。
template <typename T>
void binary(void *kv, const std::string &op, const View &va, const View &vb,
            const std::string &out) {
    auto A = borrow<T>(va.body(), va.dims());
    auto B = borrow<T>(vb.body(), vb.dims());
    auto C = alloc_out<T>(kv, out, va.kindexpr, va.dims());
    if (op == "tensor.add")
        tf::add<tf::miaobyte, T>(A, B, C);
    else if (op == "tensor.sub")
        tf::sub<tf::miaobyte, T>(A, B, C);
    else if (op == "tensor.mul")
        tf::mul<tf::miaobyte, T>(A, B, C);
    else if (op == "tensor.div") {
        if constexpr (std::is_floating_point_v<T>) // 整型 SIMD 无除法，仅浮点
            tf::div<tf::miaobyte, T>(A, B, C);
        else
            fprintf(stderr, "deepx-cpu: tensor.div 暂不支持整型\n");
    }
    if (getenv("DEEPX_DUMP")) {
        fprintf(stderr, "[deepx-cpu] %s ->", op.c_str());
        for (int i = 0; i < C.shape.size; i++)
            fprintf(stderr, " %g", (double)C.data[i]);
        fprintf(stderr, "\n");
    }
}

void dispatch(void *kv, const std::string &op, const std::string &pc) {
    std::string p0 = take(kvlang_rwirextResolveReadPath(kv, pc.c_str(), 0));
    std::string p1 = take(kvlang_rwirextResolveReadPath(kv, pc.c_str(), 1));
    std::string out = take(kvlang_rwirextResolveWrite(kv, pc.c_str(), 0));
    View va = read_view(kv, p0), vb = read_view(kv, p1);
    if (!va.found || !vb.found || out.empty()) {
        fprintf(stderr, "deepx-cpu: %s 缺参 @ %s\n", op.c_str(), pc.c_str());
        return;
    }
    std::string k = kind_of(va.kindexpr);
    if (k == "float64")
        binary<double>(kv, op, va, vb, out);
    else if (k == "float32")
        binary<float>(kv, op, va, vb, out);
    else if (k == "int64")
        binary<int64_t>(kv, op, va, vb, out);
    else if (k == "int32")
        binary<int32_t>(kv, op, va, vb, out);
    else
        fprintf(stderr, "deepx-cpu: %s 不支持 dtype %s\n", op.c_str(), k.c_str());
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
            if (!in_myrwircaps(op)) {
                stop_op = op;
                break;
            }
            dispatch(kv, op, c);
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
        if (!stop_op.empty() && in_myrwircaps(stop_op)) {
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
    register_caps(kv);

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
