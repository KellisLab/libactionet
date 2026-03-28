// Simplex regression algorithm
// Implemented from: Fast and Robust Archetypal Analysis for Representation
#ifndef ACTIONET_SIMPLEX_REGRESSION_HPP
#define ACTIONET_SIMPLEX_REGRESSION_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Run simplex regression with an active-set method.
    /// @details Solves min_X ||AX - B|| subject to simplex constraints.
    ///
    /// @param A Input matrix A in AX - B.
    /// @param B Input matrix B in AX - B.
    /// @param computeXtX If true, return X^T X (implementation-dependent).
    ///
    /// @return Solution matrix X.
    arma::mat runSimplexRegression(const arma::mat& A, const arma::mat& B, bool computeXtX = false);
} // namespace actionet

#endif //ACTIONET_SIMPLEX_REGRESSION_HPP
