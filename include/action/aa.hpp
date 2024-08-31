// Solves the standard Archetypal Analysis (AA) problem
#ifndef ACTIONET_AA_HPP
#define ACTIONET_AA_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Run archetypal analysis (AA)
    ///
    /// @param A Input matrix.
    /// @param W0 Matrix with <em>k</em> columns representing initial archetypes.
    /// @param max_it Maximum number of iterations.
    /// @param tol Convergence tolerance.
    ///
    /// @return Field with matrices <b>C</b> (<b>A.n_cols</b> x <em>k</em>) and <b>H</b> (<em>k</em> x <b>A.n_cols</b>).
    arma::field<arma::mat> run_AA(arma::mat& A, arma::mat& W0, int max_it = 100, double tol = 1e-6);
} // namespace actionet

#endif //ACTIONET_AA_HPP
