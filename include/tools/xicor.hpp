// Computes Xi correlation coefficient for vectors and matrices
// S.Chatterjee, A new coefficient of correlation (2019) (https://doi.org/10.48550/arXiv.1909.10140)
#ifndef ACTIONET_XICOR_HPP
#define ACTIONET_XICOR_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Compute Xi correlation for two vectors.
    ///
    /// Follows Chatterjee (2020) and the reference `XICOR::xicor` R implementation.
    /// Ties in Y (and -Y) are handled via `rank(., ties.method = "max") / n`.
    /// Ties in X are broken *randomly* by applying an initial joint permutation
    /// of (X, Y) driven by @p seed; this is required to preserve the asymptotic
    /// theory in the presence of ties. When @p seed == 0 the algorithm is
    /// deterministic and tied X values keep their input order.
    ///
    /// @param xvec Input vector X.
    /// @param yvec Input vector Y.
    /// @param compute_pval Compute z-score if true. `out(1)` is 0 when false.
    /// @param seed RNG seed for random tie-breaking on X. 0 = deterministic.
    /// @return Vector of statistics `(xi, z)`.
    arma::vec xicor(arma::vec xvec, arma::vec yvec, bool compute_pval = true, int seed = 0);

    /// @brief Compute Xi correlation between two matrices column-wise.
    ///
    /// Result `(i, j)` is `xicor(X.col(i), Y.col(j))`. Note that xi is
    /// asymmetric: `XICOR(X, Y) != trans(XICOR(Y, X))` in general.
    /// Per-column ranks are precomputed once and reused across all pairs,
    /// avoiding the O(nX * nY) rank recomputations of a naive nested loop.
    ///
    /// @param X Matrix X (n x p).
    /// @param Y Matrix Y (n x q). Must have the same row count as X.
    /// @param compute_pval Compute z-scores if true.
    /// @param seed RNG seed for random tie-breaking on X. 0 = deterministic.
    /// @param thread_no Number of threads (0 = auto).
    /// @return Field of two matrices: `field(0) = XI` (p x q), `field(1) = Z` (p x q).
    arma::field<arma::mat> XICOR(const arma::mat& X, const arma::mat& Y, bool compute_pval = true, int seed = 0, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_XICOR_HPP
