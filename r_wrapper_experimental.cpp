// Rcpp interface for libactionet
// [[Rcpp::interfaces(r, cpp)]]

// Enable build configuration for R interface
#define LIBACTIONET_BUILD_R
// Header `libactionet.hpp` configures package and includes `RcppArmadillo.h`. It must precede `Rcpp.h`.
#include "libactionet.hpp"
// [[Rcpp::depends(RcppArmadillo)]]

