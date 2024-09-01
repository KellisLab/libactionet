// Main ACTION decomposition
#ifndef ACTIONET_ACTION_DECOMP_HPP
#define ACTIONET_ACTION_DECOMP_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Structs
    /// @brief Stores the output of <code>run_ACTION()</code>
    ///
    /// Contains the following members:
    /// - <b>selected_cols</b>: Field of containing vectors of <b>selected_cols</b> from <code>run_SPA()</code>
    /// for each <code>k</code> in [<code>k_min</code>, <code>k_max</code>].
    /// - <b>C</b>, <b>H</b>: Fields of <b>C</b> and <b>H</b> matrices from <code>run_AA()</code>
    /// for each <code>k</code> in [<code>k_min</code>, <code>k_max</code>].
    ///
    /// @remark See <code>run_SPA()</code>, <code>ResSPA</code>, and <code>run_AA()</code>.
    struct ResACTION {
        arma::field<arma::uvec> selected_cols;
        arma::field<arma::mat> C;
        arma::field<arma::mat> H;
    };

    // Functions
    /// @brief Run ACTION decomposition algorithm
    ///
    /// @param S_r Input matrix (<em>m</em> x <em>n</em>; <em>vars</em> x <em>obs</em>).
    /// Usually a reduced representation of the raw data.
    /// @param k_min Minimum depth of decomposition (>= 2), and the beginning of the search range.
    /// @param k_max Maximum depth of decomposition (<= <code>S_r.n_cols</code>), and the end of the search range.
    /// @param normalization Normalization method to apply on <b>S_r</b> before running ACTION.
    /// @param max_it Maximum number of iterations. Passed to <code>run_AA()</code>.
    /// @param tol Convergence tolerance. Passed to <code>run_AA()</code>.
    /// @param thread_no Number of CPU threads to use. If 0, number is automatically determined.
    ///
    /// @return <code>struct</code> of type <code>ResACTION</code>.
    ///
    /// @remark <code>k</code> in [<code>k_min</code>, <code>k_max</code>] passed to <code>run_SPA(k=k)</code>.
    /// @remark See <code>ResACTION</code>, <code>run_SPA()</code>, <code>run_AA()</code>.
    ResACTION
        run_ACTION(arma::mat& S_r, int k_min, int k_max, int normalization = 0, int max_it = 100, double tol = 1e-6,
                   int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_DECOMP_HPP
