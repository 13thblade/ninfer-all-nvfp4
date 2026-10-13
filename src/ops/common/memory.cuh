#pragma once

#include "ops/common/sm70_emulation.cuh"

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

#if defined(NINFER_SM75)
// Turing has no cp.async (sm_80). Each copy goes global -> register -> shared at once, with the
// same cache policy and the same zero-fill contract, so nothing is ever outstanding and
// cp_commit/cp_wait are empty. Every caller already separates a copy from its readers with
// cp_wait and a barrier, and never writes a buffer that is still being read (an asynchronous copy
// could land at any time after it was issued), so the barriers order these plain shared stores the
// same way. Multi-stage rings become synchronous: correct, but without their latency hiding.
template <class V, Cache Policy>
__device__ __forceinline__ V sm75_global_load(const void* gmem_src) {
    if constexpr (Policy == Cache::cg) {
        return __ldcg(reinterpret_cast<const V*>(gmem_src));
    } else {
        return __ldca(reinterpret_cast<const V*>(gmem_src));
    }
}

template <int Bytes, Cache Policy>
__device__ __forceinline__ void sm75_copy(void* smem_dst, const void* gmem_src, int src_bytes) {
    using V =
        std::conditional_t<Bytes == 16, uint4, std::conditional_t<Bytes == 8, uint2, unsigned>>;
    if (src_bytes >= Bytes) {
        *reinterpret_cast<V*>(smem_dst) = sm75_global_load<V, Policy>(gmem_src);
        return;
    }
    // cp.async reads only src_bytes and zero-fills the rest; with none it reads nothing, and the
    // callers rely on that to pass an address past the end of their tensor.
    union {
        V vec;
        unsigned char bytes[Bytes];
    } staged;
    staged.vec = V{};
    const auto* src = reinterpret_cast<const unsigned char*>(gmem_src);
    for (int i = 0; i < src_bytes; ++i) staged.bytes[i] = src[i];
    *reinterpret_cast<V*>(smem_dst) = staged.vec;
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    static_assert(Policy == Cache::ca || Bytes == 16, "cp.async.cg requires a 16-byte copy");
    sm75_copy<Bytes, Policy>(smem_dst, gmem_src, Bytes);
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    static_assert(Policy == Cache::ca || Bytes == 16, "cp.async.cg requires a 16-byte copy");
    sm75_copy<Bytes, Policy>(smem_dst, gmem_src, src_bytes);
}

__device__ __forceinline__ void cp_commit() {}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
}
#else
template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
#if defined(NINFER_SM70_DEVICE)
    sm70::copy_to_shared<Bytes, Policy == Cache::cg>(smem_dst, gmem_src);
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
#if defined(NINFER_SM70_DEVICE)
    sm70::copy_to_shared_zfill<Bytes, Policy == Cache::cg>(smem_dst, gmem_src, src_bytes);
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
#endif
}

__device__ __forceinline__ void cp_commit() {
#if !defined(NINFER_SM70_DEVICE)
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if !defined(NINFER_SM70_DEVICE)
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}
#endif

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops
