// Simplex regression algorithm
// Implemented from: Fast and Robust Archetypal Analysis for Representation
#ifndef ACTIONET_SIMPLEX_REGRESSION_HPP
#define ACTIONET_SIMPLEX_REGRESSION_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Run simplex regression
    /// @details Solves min_{X} (|| AX - B ||) s.t. simplex constraint using ACTIVE Set Method
    ///
    /// @param A Input matrix <em>A</em> in <em>AX - B</em>.
    /// @param B Inout matrix <em>B</em> in <em>AX - B</em>.
    /// @param computeXtX Return <em>Xt(X)</em>
    ///
    /// @return Matrix X that solves the simplex constraint.
    arma::mat runSimplexRegression(arma::mat& A, arma::mat& B, bool computeXtX = false);
} // namespace actionet

#endif //ACTIONET_SIMPLEX_REGRESSION_HPP
