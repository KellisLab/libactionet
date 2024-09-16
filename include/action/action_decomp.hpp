// Main ACTION decomposition
#ifndef ACTIONET_ACTION_DECOMP_HPP
#define ACTIONET_ACTION_DECOMP_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Structs
    /// @brief Stores the output of <code>runACTION()</code>
    ///
    /// Contains the following members:
    /// - <b>selected_cols</b>: Field of containing vectors of <b>selected_cols</b> from <code>runSPA()</code>
    /// for each <code>k</code> in [<code>k_min</code>, <code>k_max</code>].
    /// - <b>C</b>, <b>H</b>: Fields of <b>C</b> and <b>H</b> matrices from <code>runAA()</code>
    /// for each <code>k</code> in [<code>k_min</code>, <code>k_max</code>].
    ///
    /// @remark See <code>runSPA()</code>, <code>ResSPA</code>, and <code>runAA()</code>.
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
    /// @param max_it Maximum number of iterations. Passed to <code>runAA()</code>.
    /// @param tol Convergence tolerance. Passed to <code>runAA()</code>.
    /// @param thread_no Number of CPU threads to use. If 0, number is automatically determined.
    ///
    /// @return <code>struct</code> of type <code>ResACTION</code>.
    ///
    /// @remark <code>k</code> in [<code>k_min</code>, <code>k_max</code>] passed to <code>runSPA(k=k)</code>.
    /// @remark See <code>ResACTION</code>, <code>runSPA()</code>, <code>runAA()</code>.
    ResACTION
        runACTION(arma::mat& S_r, int k_min, int k_max, int max_it = 100, double tol = 1e-6,
                  int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_DECOMP_HPP
