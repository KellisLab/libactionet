#ifndef ACTIONET_AUTOCORRELATION_HPP
#define ACTIONET_AUTOCORRELATION_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Moran's I autocorrelation (parametric).
    ///
    /// @param G Symmetric adjacency matrix.
    /// @param scores Nodes x features matrix.
    /// @param normalization_method Normalization method: 0=none, 1=zscore, 2=robust_zscore, 3=mean_center.
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field with statistic and p-values.
    arma::field<arma::vec>
        autocorrelation_Moran_parametric(const arma::sp_mat& G, const arma::mat& scores, int normalization_method = 3,
                                         int thread_no = 0);

    /// @brief Moran's I autocorrelation via permutation.
    ///
    /// @param G Symmetric adjacency matrix.
    /// @param scores Nodes x features matrix.
    /// @param normalization_method Normalization method: 0=none, 1=zscore, 2=robust_zscore, 3=mean_center.
    /// @param perm_no Number of permutations.
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field with statistic and p-values.
    arma::field<arma::vec>
        autocorrelation_Moran(const arma::sp_mat& G, const arma::mat& scores, int normalization_method = 1, int perm_no = 30,
                              int thread_no = 0);

    /// @brief Geary's C autocorrelation via permutation.
    ///
    /// @param G Symmetric adjacency matrix.
    /// @param scores Nodes x features matrix.
    /// @param normalization_method Normalization method: 0=none, 1=zscore, 2=robust_zscore, 3=mean_center.
    /// @param perm_no Number of permutations.
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field with statistic and p-values.
    arma::field<arma::vec>
        autocorrelation_Geary(const arma::sp_mat& G, const arma::mat& scores, int normalization_method = 1,
                              int perm_no = 30, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_AUTOCORRELATION_HPP
