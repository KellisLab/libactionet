// Extension to hnsw for computing Jensen-Shannon distance.
//
// Design notes (2026 rewrite):
//   The previous implementation precomputed a 1,000,002-entry (~4 MB) lookup
//   table of fasterlog2(i/LOGLEN) and, for every element of every distance
//   evaluation, performed three data-dependent gathers into that table plus
//   three floor() calls and three branches.  JSD is the default network metric
//   and this function is the single hottest instruction stream in
//   build_network, so the table's cache footprint and the gather/branch
//   dependency chains dominated runtime and blocked vectorization.
//
//   This version computes fasterlog2 directly per element via a memcpy-based
//   bit reinterpret.  fasterlog2 is a handful of cheap FLOPs (a reinterpret, a
//   multiply, a subtract), so dropping the table removes ~4 MB of per-space
//   cache pressure, removes the gather dependency chains, and lets the compiler
//   auto-vectorize the loop on every supported target (NEON on arm64, SSE/AVX
//   on x86) under -O3.  Computing the log directly is also slightly *more*
//   accurate than the quantized table, so results are not bitwise identical to
//   the old kernel; parity is asserted within a documented tolerance by tests
//   rather than bit-for-bit.
#ifndef ACTIONET_HNSW_JENSEN_SHANNON_HPP
#define ACTIONET_HNSW_JENSEN_SHANNON_HPP
#include <cmath>
#include <cstdint>
#include <cstring>

namespace hnswlib {

    // Vectorizer-friendly base-2 log approximation (Mineiro's fasterlog2).
    // Uses a memcpy bit reinterpret instead of the fastapprox union type-pun so
    // the compiler can auto-vectorize the enclosing loop (NEON on arm64, SSE on
    // x86); union punning inside the hot loop blocks clang's loop vectorizer.
    // Arithmetic is identical to fastapprox::fasterlog2.
    static inline float jsd_fasterlog2(float x) {
        std::uint32_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        return static_cast<float>(bits) * 1.1920928955078125e-7f - 126.94269504f;
    }

    // x * log2(x), defined as 0 for x <= 0.  The (x > 0) predicate lowers to a
    // vector select/mask, so the product is exactly 0 for x <= 0 without a
    // data-dependent branch, preserving the original kernel's "p == 0 ? 0"
    // semantics while remaining vectorizable.
    static inline float jsd_xlog2x(float x) {
        const float v = x * jsd_fasterlog2(x);
        return (x > 0.0f) ? v : 0.0f;
    }

    static float computeJSDMetric(const void* pVect1_p, const void* pVect2_p,
                            const void* params) {
        const std::size_t N = *static_cast<const std::size_t*>(params);

        const float* __restrict pVect1 = static_cast<const float*>(pVect1_p);
        const float* __restrict pVect2 = static_cast<const float*>(pVect2_p);

        float sum1 = 0.0f, sum2 = 0.0f;
        float sum_p = 0.0f, sum_q = 0.0f;
        #if defined(__clang__)
        #pragma clang loop vectorize(enable)
        #endif
        for (std::size_t i = 0; i < N; i++) {
            const float p = pVect1[i];
            const float q = pVect2[i];
            const float m = 0.5f * (p + q);

            sum_p += p;
            sum_q += q;

            sum1 += jsd_xlog2x(p) + jsd_xlog2x(q);
            sum2 += jsd_xlog2x(m);
        }

        const float res1 = 1.0f - sum_p;
        const float res2 = 1.0f - sum_q;

        float JS = 0.5f * sum1 - sum2;
        JS = 0.0f < JS ? JS : 0.0f;
        JS += 0.5f * ((0.0f < res1 ? res1 : 0.0f) + (0.0f < res2 ? res2 : 0.0f));

        return std::sqrt(JS);
    }

    class JSDSpace : public SpaceInterface<float> {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        JSDSpace(size_t dim) {
            fstdistfunc_ = computeJSDMetric;
            data_size_ = dim * sizeof(float);
            dim_ = dim;
        }

        size_t get_data_size() { return data_size_; }

        DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

        // The distance-function parameter is now simply the dimensionality.
        void* get_dist_func_param() { return &dim_; }

        ~JSDSpace() {}
    };
} // namespace hnswlib

#endif //ACTIONET_HNSW_JENSEN_SHANNON_HPP
