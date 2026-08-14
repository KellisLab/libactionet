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
//   auto-vectorize the loop (NEON on arm64, SSE/AVX on x86).
//
//   Vectorization portability (2026-08 GCC audit): the loop is annotated with
//   `#pragma omp simd` (OpenMP is a hard build requirement) rather than a
//   clang-only loop hint.  Under GCC -- the default manylinux/HPC compiler --
//   the loop would NOT auto-vectorize at -O3 or even -march=native without
//   this: GCC bailed with "control flow in loop" on the (x > 0) ? v : 0
//   select that wrapped the bit-reinterpret log.  The zero-guard is therefore
//   written as a 1.0f/0.0f mask multiply (see jsd_xlog2x), which GCC and Clang
//   both lower to a packed blend, yielding SSE (16-byte) vectors on the default
//   x86 build and AVX2/AVX-512 (32/64-byte) vectors under install_optimized.sh.
//
//   Computing the log directly is also slightly *more* accurate than the
//   quantized table, so results are not bitwise identical to the old kernel;
//   parity is asserted within a documented tolerance by tests rather than
//   bit-for-bit.
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

    // x * log2(x), defined as 0 for x <= 0.  The zero-guard is expressed as a
    // multiply by a 1.0f/0.0f mask rather than a select around the whole
    // expression.  This matters for portability: GCC (the default
    // manylinux/HPC toolchain) refuses to vectorize the loop when a select is
    // wrapped around the bit-reinterpret log (it reports "control flow in
    // loop" and falls back to scalar), whereas the mask multiply lowers to a
    // packed compare + blend that GCC and Clang both vectorize.  The result is
    // bit-identical to the (x > 0) ? v : 0 form under scalar evaluation and
    // preserves the original kernel's "p == 0 ? 0" semantics (for x == 0 the
    // reinterpret log is finite, so 0 * log * 0 == 0; no inf/nan can leak
    // through the mask).
    static inline float jsd_xlog2x(float x) {
        const float mask = (x > 0.0f) ? 1.0f : 0.0f;
        return (x * jsd_fasterlog2(x)) * mask;
    }

    static float computeJSDMetric(const void* pVect1_p, const void* pVect2_p,
                            const void* params) {
        const std::size_t N = *static_cast<const std::size_t*>(params);

        const float* __restrict pVect1 = static_cast<const float*>(pVect1_p);
        const float* __restrict pVect2 = static_cast<const float*>(pVect2_p);

        float sum1 = 0.0f, sum2 = 0.0f;
        float sum_p = 0.0f, sum_q = 0.0f;
        // OpenMP is a hard build requirement, so `omp simd` is the portable way
        // to force vectorization of this reduction across GCC and Clang.  The
        // older clang-only `#pragma clang loop vectorize(enable)` hint is kept
        // as a fallback for the (unsupported) no-OpenMP clang build.
        #if defined(_OPENMP)
        #pragma omp simd reduction(+ : sum1, sum2, sum_p, sum_q)
        #elif defined(__clang__)
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
