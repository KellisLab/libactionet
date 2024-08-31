// Main ACTION decomposition
#ifndef ACTIONET_ACTION_DECOMP_HPP
#define ACTIONET_ACTION_DECOMP_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Structs
    /// @brief Stores the output of \c run_ACTION()
    ///
    /// Contains the following members:
    /// - <b>selected_cols</b>: Field containing vectors of <b>selected_cols</b> from <code>run_SPA()</code>
    /// for each <em>k</em> in [<code>k_min</code>, <code>k_max</code>].
    /// - <b>C</b>, <b>H</b>: Fields of <b>C</b> and <b>H</b> matrices from <code>run_AA()</code>
    /// for each <em>k</em> in [<code>k_min</code>, <code>k_max</code>].
    ///
    /// See <code>run_SPA()</code>, <code>ResSPA</code>, and <code>run_AA()</code>.
    struct ResACTION {
        arma::field<arma::uvec> selected_cols;
        arma::field<arma::mat> C;
        arma::field<arma::mat> H;
    };

    // Functions
    /// @brief Run ACTION decomposition algorithm
    ///
    /// @param S_r Input matrix. Usually a reduced representation of the raw data.
    /// @param k_min Minimum number of archetypes (>= 2) to search for, and the beginning of the search range.
    /// @param k_max Maximum number of archetypes (<= <b>S_r.n_cols</b>) to search for, and the end of the search range.
    /// @param normalization Normalization method to apply on <b>S_r</b> before running ACTION.
    /// @param max_it Maximum number of iterations for <code>run_AA()</code>.
    /// @param tol Convergence tolerance for <code>run_AA()</code>.
    /// @param thread_no Number of CPU threads to use. If 0, number is automatically determined.
    ///
    /// @return <code>struct</code> of type <code>ResACTION</code>.
    ///
    /// See <code>ResACTION</code>.
    ResACTION
        run_ACTION(arma::mat& S_r, int k_min, int k_max, int normalization = 0, int max_it = 100, double tol = 1e-6,
                   int thread_no = 0);
} // namespace actionet


#endif //ACTIONET_ACTION_DECOMP_HPP
