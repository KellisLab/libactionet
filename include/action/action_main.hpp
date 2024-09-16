// Main ACTION decomposition
#ifndef ACTIONET_ACTION_MAIN_HPP
#define ACTIONET_ACTION_MAIN_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
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
    arma::field<arma::mat> runACTION(arma::mat& S_r, int k_min, int k_max, int max_it = 100, double tol = 1e-6,
                                     double spec_th = -3, int min_obs = 3, int norm = 0, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_MAIN_HPP
