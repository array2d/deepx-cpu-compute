#pragma once
// kvspace body ↔ deepx::Tensor<T> 零拷贝桥。
// 读：kvspaceGet 借用指针 → DecodeHead 取 dims/langtype → 原位建 Tensor<T>（借用 data，deleter=nullptr）。
// 写：kvspaceWriteNewPlace 要可写 body 偏移指针 → 原位建 Tensor<T> → kernel 直接写 kvspace body。

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <kvspace/kvspace.h>

#include "deepx/tensor.hpp"

namespace deepx::rt {

constexpr uint8_t STORETYPE_ARRAYND = 2;

// langtype（如 "[2,3]float32"）剥 [dims] 前缀取基础 dtype 名。
inline std::string kind_of(const std::string &langtype) {
    if (!langtype.empty() && langtype[0] == '[') {
        auto e = langtype.find(']');
        if (e != std::string::npos)
            return langtype.substr(e + 1);
    }
    return langtype;
}

// 把 langtype（如 "[2,3]float32"）的 dtype 换成 kind，保留 [dims] 前缀（如 → "[2,3]bool"）。
inline std::string with_kind(const std::string &langtype, const std::string &kind) {
    if (!langtype.empty() && langtype[0] == '[') {
        auto e = langtype.find(']');
        if (e != std::string::npos)
            return langtype.substr(0, e + 1) + kind;
    }
    return kind;
}

// 由 dims + dtype 名拼 langtype（如 {2,3},"float32" → "[2,3]float32"）；输出形状与输入不同的算子（matmul/reduce/reshape）用。
inline std::string make_langtype(const std::vector<int> &dims, const std::string &kind) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); i++) {
        if (i)
            s += ",";
        s += std::to_string(dims[i]);
    }
    s += "]";
    s += kind;
    return s;
}

inline int elem_size(const std::string &kind) {
    if (kind == "float64" || kind == "int64" || kind == "uint64")
        return 8;
    if (kind == "float32" || kind == "int32" || kind == "uint32")
        return 4;
    if (kind == "int16" || kind == "uint16")
        return 2;
    if (kind == "int8" || kind == "uint8" || kind == "bool")
        return 1;
    return 0;
}

// 读参路径的 head（借用指针 + 元信息）。found=false 表示无值。
struct View {
    uint8_t *base = nullptr; // kvspaceGet 借用指针（不得 free）
    kvspaceHead_t head{};
    std::string langtype;
    bool found = false;
    void *body() const { return base + head.body_offset; }
    std::vector<int> dims() const {
        std::vector<int> d;
        for (int i = 0; i < head.ndim; i++)
            d.push_back(head.dims[i]);
        return d;
    }
    int numel() const {
        int n = 1;
        for (int i = 0; i < head.ndim; i++)
            n *= head.dims[i];
        return head.ndim == 0 ? 1 : n;
    }
};

inline View read_view(void *kv, const std::string &path) {
    View v;
    uint8_t *out = nullptr;
    uint32_t olen = 0;
    if (kvspaceGet(kv, path.c_str(), 1, &out, &olen) != 0 || !out || olen == 0)
        return v;
    if (kvspaceDecodeHead(out, olen, &v.head) != 0)
        return v;
    v.base = out;
    v.langtype.assign((const char *)v.head.langtype,
                      strnlen((const char *)v.head.langtype, sizeof(v.head.langtype)));
    v.found = true;
    return v;
}

// memcpy 版 copyer（CopyFn 语义 (src, dst, n)），供 reshape 等需拷贝的 kernel 用；只读 src 写 dst，安全。
template <typename T>
inline void mem_copy(T *src, T *dst, int n) {
    std::memcpy(dst, src, (size_t)n * sizeof(T));
}

// 在借用/可写 body 指针上原位建 Tensor<T>（不拥有内存：deleter/newer=nullptr，绝不 free kvspace body）。
template <typename T>
inline Tensor<T> borrow(void *data, const std::vector<int> &dims) {
    Tensor<T> t;
    t.shape = Shape(dims);
    t.data = (T *)data;
    t.deleter = nullptr;
    t.newer = nullptr;
    t.copyer = &mem_copy<T>;
    return t;
}

// 分配输出：WriteNewPlace 要可写 body 偏移指针，返回借用 Tensor<T>（写入即写 kvspace）。
template <typename T>
inline Tensor<T> alloc_out(void *kv, const std::string &key, const std::string &langtype,
                           const std::vector<int> &dims) {
    int n = 1;
    for (int d : dims)
        n *= d;
    if (dims.empty())
        n = 1;
    uint8_t *bp = nullptr;
    char err[256] = {0};
    kvspaceWriteNewPlace(kv, key.c_str(), 0, STORETYPE_ARRAYND, 0, 0, langtype.c_str(),
                         (uint32_t)(n * sizeof(T)), &bp, err, sizeof(err));
    return borrow<T>(bp, dims);
}

} // namespace deepx::rt
