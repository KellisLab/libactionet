// Main ACTION decomposition
#ifndef ACTIONET_ACTION_MAIN_HPP
#define ACTIONET_ACTION_MAIN_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Run ACTION decomposition with pruning and merging.
    ///
    /// @param S_r Reduced input matrix (<em>vars</em> x <em>obs</em>).
    /// @param k_min Minimum number of archetypes (>= 2).
    /// @param k_max Maximum number of archetypes (<= <code>S_r.n_cols</code>).
    /// @param max_it Maximum number of iterations for AA.
    /// @param tol Convergence tolerance for AA.
    /// @param spec_th Specificity threshold (z-score) for pruning.
    /// @param min_obs Minimum observations per archetype.
    /// @param thread_no Number of CPU threads (0 = auto).
    ///
    /// @return Field of matrices containing stacked and merged C/H results.
    arma::field<arma::mat> runACTION(arma::mat& S_r, int k_min, int k_max, int max_it = 100, double tol = 1e-6,
                                     double spec_th = -3, int min_obs = 3, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_MAIN_HPP
