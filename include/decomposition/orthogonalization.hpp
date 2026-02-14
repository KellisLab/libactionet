// Orthogonalization method for batch correction
#ifndef ACTIONET_ORTHOGONALIZATION_HPP
#define ACTIONET_ORTHOGONALIZATION_HPP

#include "libactionet_config.hpp"

// Functions: internal
/// @brief Deflate a reduced representation using perturbation matrices.
///
/// @param SVD_results SVD field output (modified in place by perturbedSVD).
/// @param A Perturbation matrix A.
/// @param B Perturbation matrix B.
///
/// @return Updated SVD field.
arma::field<arma::mat> deflateReduction(arma::field<arma::mat>& SVD_results, const arma::mat& A, const arma::mat& B);

namespace actionet {
    /// @brief Orthogonalize a reduced representation against a batch design matrix.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param S Original input matrix.
    /// @param SVD_results SVD results from reduction.
    /// @param design Batch design matrix (cells x covariates).
    ///
    /// @return Corrected SVD field.
    template <typename T>
    arma::field<arma::mat> orthogonalizeBatchEffect(T& S, arma::field<arma::mat>& SVD_results, arma::mat& design);

    /// @brief Orthogonalize a reduced representation against a basal expression profile.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param S Original input matrix.
    /// @param SVD_results SVD results from reduction.
    /// @param basal_state Basal expression vector/matrix.
    ///
    /// @return Corrected SVD field.
    template <typename T>
    arma::field<arma::mat> orthogonalizeBasal(T& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);
} // namespace actionet

#endif //ACTIONET_ORTHOGONALIZATION_HPP
