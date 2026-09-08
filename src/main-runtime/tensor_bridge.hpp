#pragma once
// kvspace body ↔ deepx::Tensor<T> 零拷贝桥。
// 读：kvspaceGet 借用指针 → DecodeHead 取 dims/kindexpr → 原位建 Tensor<T>（借用 data，deleter=nullptr）。
// 写：kvspaceWriteNewPlace 要可写 body 偏移指针 → 原位建 Tensor<T> → kernel 直接写 kvspace body。

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <kvspace/kvspace.h>

#include "deepx/tensor.hpp"

namespace deepx::rt {

constexpr uint8_t STORETYPE_ARRAYND = 2;

// kindexpr（如 "[2,3]float32"）剥 [dims] 前缀取基础 dtype 名。
inline std::string kind_of(const std::string &kindexpr) {
    if (!kindexpr.empty() && kindexpr[0] == '[') {
        auto e = kindexpr.find(']');
        if (e != std::string::npos)
            return kindexpr.substr(e + 1);
    }
    return kindexpr;
}

// 把 kindexpr（如 "[2,3]float32"）的 dtype 换成 kind，保留 [dims] 前缀（如 → "[2,3]bool"）。
inline std::string with_kind(const std::string &kindexpr, const std::string &kind) {
    if (!kindexpr.empty() && kindexpr[0] == '[') {
        auto e = kindexpr.find(']');
        if (e != std::string::npos)
            return kindexpr.substr(0, e + 1) + kind;
    }
    return kind;
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
    std::string kindexpr;
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
    v.kindexpr.assign((const char *)v.head.langtype,
                      strnlen((const char *)v.head.langtype, sizeof(v.head.langtype)));
    v.found = true;
    return v;
}

// 在借用/可写 body 指针上原位建 Tensor<T>（不拥有内存：deleter=nullptr）。
template <typename T>
inline Tensor<T> borrow(void *data, const std::vector<int> &dims) {
    Tensor<T> t;
    t.shape = Shape(dims);
    t.data = (T *)data;
    t.deleter = nullptr;
    t.newer = nullptr;
    t.copyer = nullptr;
    return t;
}

// 分配输出：WriteNewPlace 要可写 body 偏移指针，返回借用 Tensor<T>（写入即写 kvspace）。
template <typename T>
inline Tensor<T> alloc_out(void *kv, const std::string &key, const std::string &kindexpr,
                           const std::vector<int> &dims) {
    int n = 1;
    for (int d : dims)
        n *= d;
    if (dims.empty())
        n = 1;
    uint8_t *bp = nullptr;
    char err[256] = {0};
    kvspaceWriteNewPlace(kv, key.c_str(), 0, STORETYPE_ARRAYND, 0, 0, kindexpr.c_str(),
                         (uint32_t)(n * sizeof(T)), &bp, err, sizeof(err));
    return borrow<T>(bp, dims);
}

} // namespace deepx::rt
