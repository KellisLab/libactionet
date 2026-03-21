// Orthogonalization method for batch correction
#ifndef ACTIONET_ORTHOGONALIZATION_HPP
#define ACTIONET_ORTHOGONALIZATION_HPP

#include "libactionet_config.hpp"
#include "decomposition/svd_main.hpp"

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

    // ---- Operator-backed orthogonalization ------------------------------------------------

    /// @brief Orthogonalize a reduced representation against a batch design matrix
    ///        using a MatrixOperator for the backed matrix products.
    ///
    /// Computes Z = orth(S * design), B = -(Z' * S)', then deflates the existing
    /// SVD with mean-augmented perturbation terms via perturbedSVD.
    ///
    /// @param S       Matrix operator (features x cells).
    /// @param svd     Current SVD decomposition.
    /// @param prior   Previously accumulated perturbation (may be nullptr).
    /// @param design  Batch design matrix (cells x covariates).
    ///
    /// @return Updated SVD with accumulated perturbation terms.
    PerturbedSVDResult orthogonalizeBatchEffect_Operator(
        const MatrixOperator& S,
        const SVDResult& svd,
        const PerturbedSVDResult* prior,
        const arma::mat& design);

    /// @brief Orthogonalize a reduced representation against a basal expression
    ///        profile using a MatrixOperator for the backed matrix products.
    ///
    /// Computes Z = orth(basal_state), B = -(Z' * S)', then deflates via perturbedSVD.
    ///
    /// @param S           Matrix operator (features x cells).
    /// @param svd         Current SVD decomposition.
    /// @param prior       Previously accumulated perturbation (may be nullptr).
    /// @param basal_state Basal expression vector/matrix (features x q).
    ///
    /// @return Updated SVD with accumulated perturbation terms.
    PerturbedSVDResult orthogonalizeBasal_Operator(
        const MatrixOperator& S,
        const SVDResult& svd,
        const PerturbedSVDResult* prior,
        const arma::mat& basal_state);

} // namespace actionet

#endif //ACTIONET_ORTHOGONALIZATION_HPP
