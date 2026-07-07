// Main ACTION decomposition
#ifndef ACTIONET_ACTION_DECOMP_HPP
#define ACTIONET_ACTION_DECOMP_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Structs
    /// @brief Stores the output of <code>decompACTION()</code>.
    ///
    /// Contains the following members:
    /// - <b>selected_cols</b>: Selected columns from <code>runSPA()</code> for each k.
    /// - <b>C_stacked</b>: All per-k C matrices column-stacked into a single dense matrix
    ///   (n_cells x T, where T = sum(k_min..k_max)).
    /// - <b>H_stacked</b>: All per-k H matrices row-stacked into a single dense matrix
    ///   (T x n_cells).
    struct ResACTION {
        arma::field<arma::uvec> selected_cols;
        arma::mat C_stacked; // n_cells × T
        arma::mat H_stacked; // T × n_cells
    };

    /// @brief Run ACTION decomposition across a k-range without post-processing.
    ///
    /// @param S_r Reduced input matrix (<em>k</em> × <em>cells</em>).  This is the internal
    ///            orientation used throughout the ACTION decomposition; Python bindings
    ///            transpose at the boundary (see <code>wp_action.cpp</code>).
    /// @param k_min Minimum number of archetypes (>= 2).
    /// @param k_max Maximum number of archetypes (<= <code>S_r.n_cols</code>).
    /// @param max_it Maximum number of iterations for AA.
    /// @param tol Convergence tolerance for AA.
    /// @param thread_no Number of CPU threads (0 = auto).
    ///
    /// @return <code>ResACTION</code> with SPA selections and pre-stacked C/H matrices.
    ResACTION
        decompACTION(const arma::mat& S_r, int k_min, int k_max, int max_it = 100, double tol = 1e-16,
                  int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_DECOMP_HPP
