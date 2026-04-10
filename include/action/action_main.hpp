// Main ACTION decomposition
#ifndef ACTIONET_ACTION_MAIN_HPP
#define ACTIONET_ACTION_MAIN_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Run ACTION decomposition with pruning and merging.
    ///
    /// AnnData-native orientation contract (Plan 02):
    ///   Input  S_r: cells × k  (obs × archetypes-dim; was k × cells).
    ///   Output field(5):
    ///     (0) H_stacked: cells × archetypes  (was archetypes × cells)
    ///     (1) C_stacked: cells × archetypes  (unchanged)
    ///     (2) H_merged:  cells × archetypes  (was archetypes × cells)
    ///     (3) C_merged:  cells × archetypes  (unchanged)
    ///     (4) assigned_archetypes: (cells,)  (unchanged)
    ///
    /// Internally the pipeline transposes S_r to k × cells before passing it
    /// to the column-oriented SPA/AA/simplex routines, then transposes the H
    /// outputs back to cells × archetypes before returning.
    ///
    /// @param S_r Reduced input matrix (cells × k).
    /// @param k_min Minimum number of archetypes (>= 2).
    /// @param k_max Maximum number of archetypes (<= <code>S_r.n_rows</code>).
    /// @param max_it Maximum number of iterations for AA.
    /// @param tol Convergence tolerance for AA.
    /// @param spec_th Specificity threshold (z-score) for pruning.
    /// @param min_obs Minimum observations per archetype.
    /// @param thread_no Number of CPU threads (0 = auto).
    /// @param return_c_matrices Whether to retain and return C_stacked/C_merged
    /// in the output field. If false, slots (1) and (3) are returned as empty
    /// matrices while preserving field indices.
    ///
    /// @return Field of matrices containing stacked and merged C/H results.
    arma::field<arma::mat> runACTION(const arma::mat& S_r, int k_min, int k_max, int max_it = 100, double tol = 1e-6,
                                     double spec_th = -3, int min_obs = 3, int thread_no = 0,
                                     bool return_c_matrices = true);
} // namespace actionet

#endif //ACTIONET_ACTION_MAIN_HPP
