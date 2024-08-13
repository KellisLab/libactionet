#ifndef LIBACTIONET_CONFIG_HPP
#define LIBACTIONET_CONFIG_HPP

// Armadillo build configuration
#define ARMA_DONT_USE_WRAPPER
#undef ARMA_BLAS_CAPITALS
#define ARMA_BLAS_UNDERSCORE

// Statslib build configuration
#define STATS_ENABLE_ARMA_WRAPPERS 1
#define STATS_GO_INLINE

//#define ACTIONET_BUILD_R

// Platform specific headers and macros
#if defined(ACTIONET_BUILD_R)

    #define stdout_printf Rprintf
    #define stderr_printf REprintf
    #define FLUSH R_FlushConsole()

    #include <Rinterface.h>

    // Use RcppArmadillo for StatsLib
    #define USE_RCPP_ARMADILLO 1

    #include <RcppArmadillo.h>

#else

    #define stdout_printf printf
    #define stderr_printf printf
    #define FLUSH fflush(stdout)

    #include "armadillo"

#endif


#endif //LIBACTIONET_CONFIG_HPP
