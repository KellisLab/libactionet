// Computes Xi correlation coefficient for vectors and matrices
// S.Chatterjee, A new coefficient of correlation (2019) (https://doi.org/10.48550/arXiv.1909.10140)
#ifndef ACTIONET_XICOR_HPP
#define ACTIONET_XICOR_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Compute Xi correlation for two vectors.
    ///
    /// @param xvec Input vector X.
    /// @param yvec Input vector Y.
    /// @param compute_pval Compute p-value if true.
    /// @param seed Random seed.
    /// @return Vector of statistics (xi, p-value, z).
    arma::vec xicor(arma::vec xvec, arma::vec yvec, bool compute_pval = true, int seed = 0);

    /// @brief Compute Xi correlation between two matrices.
    ///
    /// @param X Matrix X (n x p).
    /// @param Y Matrix Y (n x q).
    /// @param compute_pval Compute p-values if true.
    /// @param seed Random seed.
    /// @param thread_no Number of threads (0 = auto).
    /// @return Field with matrices of statistics.
    arma::field<arma::mat> XICOR(arma::mat& X, arma::mat& Y, bool compute_pval = true, int seed = 0, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_XICOR_HPP
