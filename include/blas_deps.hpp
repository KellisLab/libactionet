#ifndef ACTIONET_BLAS_DEPS_HPP
#define ACTIONET_BLAS_DEPS_HPP

// Minimal, source-only BLAS include shim. Keep CBLAS headers out of public
// interfaces; only translation units that call BLAS should include this.
#if defined(LIBACTIONET_BLAS_MKL)
    #include <mkl_cblas.h>
#elif defined(LIBACTIONET_BLAS_ACCELERATE)
    // Accelerate provides CBLAS via vecLib.
    #include <cblas.h>
#else
    #include <cblas.h>
#endif

#endif // ACTIONET_BLAS_DEPS_HPP
