// Simplex regression algorithm
// Implemented from: Fast and Robust Archetypal Analysis for Representation
#ifndef ACTIONET_SIMPLEX_REGRESSION_HPP
#define ACTIONET_SIMPLEX_REGRESSION_HPP

#include "libactionet_config.hpp"

// Exported
namespace ACTIONet {

    // Simplex regression for AA: min_{X} (|| AX - B ||) s.t. simplex constraint using ACTIVE Set Method
    arma::mat run_simplex_regression(arma::mat &A, arma::mat &B, bool computeXtX = false);

} // namespace ACTIONet

#endif //ACTIONET_SIMPLEX_REGRESSION_HPP
