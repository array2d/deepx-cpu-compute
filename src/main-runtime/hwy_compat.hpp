#pragma once
// deepx-core 的 SIMD kernel 依赖 highway 的 IsAligned(tag, ptr)，系统 highway 1.0.7 未提供。
// 在 kernel 所用的 HWY_NAMESPACE 内补一个等价实现（ADL 由 ScalableTag<T> 命中），不改 deepx-core。

#include <cstdint>

#include <hwy/highway.h>

namespace hwy {
namespace HWY_NAMESPACE {
template <class D>
HWY_INLINE bool IsAligned(D d, const TFromD<D> *p) {
    return reinterpret_cast<uintptr_t>(p) % (Lanes(d) * sizeof(TFromD<D>)) == 0;
}
// reduce_miaobyte 用标量返回的 ReduceMax/ReduceMin，本 highway 目标未提供（仅 ReduceSum 全覆盖）；
// 经全覆盖的 MaxOfLanes/MinOfLanes + GetLane 等价补齐。
template <class D, class V>
HWY_INLINE TFromD<D> ReduceMax(D d, V v) {
    return GetLane(MaxOfLanes(d, v));
}
template <class D, class V>
HWY_INLINE TFromD<D> ReduceMin(D d, V v) {
    return GetLane(MinOfLanes(d, v));
}
} // namespace HWY_NAMESPACE
} // namespace hwy
