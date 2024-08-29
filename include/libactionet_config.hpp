#ifndef ACTIONET_CONFIG_HPP
#define ACTIONET_CONFIG_HPP

// Armadillo build configuration
#define ARMA_DONT_USE_WRAPPER
#undef ARMA_BLAS_CAPITALS
#define ARMA_BLAS_UNDERSCORE
//#define ARMA_64BIT_WORD
//#define ARMA_BLAS_LONG_LONG

// StatsLib build configuration
#define STATS_ENABLE_ARMA_WRAPPERS
#define STATS_GO_INLINE

//#define LIBACTIONET_BUILD_R // For testing

// Configurations for R and Python interface
#if defined(LIBACTIONET_BUILD_R)

    #include <Rinterface.h>
    #define stdout_printf Rprintf
    #define stderr_printf REprintf
    #define FLUSH R_FlushConsole()

    // Use RcppArmadillo for StatsLib
    #define USE_RCPP_ARMADILLO

    // #define ARMA_32BIT_WORD // Automatically set by RcppArmadillo
    #include <RcppArmadillo.h>

#else

// TODO: stdio macros
#define stdout_printf printf
#define stderr_printf printf
#define FLUSH fflush(stdout)

#include "armadillo"

#endif

// Platform specific headers and macros
#if defined(LIBACTIONET_BLAS_MKL)
    #include <mkl_cblas.h>
#elif defined(LIBACTIONET_BLAS_ACCELERATE)
    //  #include <Accelerate.h>
    #include <cblas.h>
#else
    #include <cblas.h>
#endif

#endif //ACTIONET_CONFIG_HPP
