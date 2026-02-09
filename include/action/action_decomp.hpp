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
    /// - <b>C</b>, <b>H</b>: C/H matrices from <code>runAA()</code> for each k.
    struct ResACTION {
        arma::field<arma::uvec> selected_cols;
        arma::field<arma::mat> C;
        arma::field<arma::mat> H;
    };

    /// @brief Run ACTION decomposition across a k-range without post-processing.
    ///
    /// @param S_r Reduced input matrix (<em>vars</em> x <em>obs</em>).
    /// @param k_min Minimum number of archetypes (>= 2).
    /// @param k_max Maximum number of archetypes (<= <code>S_r.n_cols</code>).
    /// @param max_it Maximum number of iterations for AA.
    /// @param tol Convergence tolerance for AA.
    /// @param thread_no Number of CPU threads (0 = auto).
    ///
    /// @return <code>ResACTION</code> with SPA selections and C/H traces.
    ResACTION
        decompACTION(arma::mat& S_r, int k_min, int k_max, int max_it = 100, double tol = 1e-6,
                  int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_DECOMP_HPP
