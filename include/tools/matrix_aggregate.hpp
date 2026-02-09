#ifndef ACTIONET_MATRIX_MISC_HPP
#define ACTIONET_MATRIX_MISC_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Compute grouped sums along an axis (sparse input).
    arma::mat computeGroupedSums(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
    /// @brief Compute grouped sums as sparse output (sparse input).
    arma::sp_mat computeGroupedSums2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);

    /// @brief Compute grouped sums along an axis (dense input).
    arma::mat computeGroupedSums(arma::mat& S, arma::vec& sample_assignments, int axis = 0);

    /// @brief Compute grouped means along an axis (dense or sparse input).
    template <typename T>
    arma::mat computeGroupedMeans(T& S, arma::vec& sample_assignments, int axis = 0);
    /// @brief Compute grouped means as sparse output (sparse input).
    arma::sp_mat computeGroupedMeans2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);

    /// @brief Compute grouped variances along an axis (sparse input).
    arma::mat computeGroupedVars(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
    /// @brief Compute grouped variances along an axis (dense input).
    arma::mat computeGroupedVars(arma::mat& S, arma::vec& sample_assignments, int axis = 0);
    /// @brief Compute grouped variances as sparse output (sparse input).
    arma::sp_mat computeGroupedVars2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
} // namespace actionet

#endif //ACTIONET_MATRIX_MISC_HPP
