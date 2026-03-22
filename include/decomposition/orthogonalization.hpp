// Orthogonalization method for batch correction
#ifndef ACTIONET_ORTHOGONALIZATION_HPP
#define ACTIONET_ORTHOGONALIZATION_HPP

#include "libactionet_config.hpp"
#include "action/reduce_kernel.hpp"

// Functions: internal
/// @brief Deflate a reduced representation using perturbation matrices.
///
/// The field layout is the public reduction contract
/// {S_r, sigma, U, A, B} returned by reduceKernel():
///   - S_r: cells x k
///   - U  : genes x k
///   - A  : genes x p
///   - B  : cells x p
///
/// @param reduction_results Reduction field in public Plan 02 layout.
/// @param A Gene-space perturbation (genes x q).
/// @param B Cell-space perturbation (cells x q).
///
/// @return Updated reduction field in the same public layout.
arma::field<arma::mat> deflateReduction(arma::field<arma::mat>& reduction_results,
                                        const arma::mat& A, const arma::mat& B);

namespace actionet {
    /// @brief Orthogonalize a reduced representation against a batch design matrix.
    ///
    /// AnnData-native orientation (Plan 02): S is cells × genes.
    ///   Z = orth(S.t() * design)  (genes × q — gene-space projection)
    ///   B = -(S * Z)              (cells × q — direct product, no extra transpose)
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param S Original input matrix (cells × genes).
    /// @param reduction_results Reduction field from reduceKernel(), in public
    ///                          Plan 02 layout {S_r, sigma, U, A, B}.
    /// @param design Batch design matrix (cells × covariates).
    ///
    /// @return Corrected reduction field in the same public layout.
    template <typename T>
    arma::field<arma::mat> orthogonalizeBatchEffect(T& S, arma::field<arma::mat>& reduction_results,
                                                    arma::mat& design);

    /// @brief Orthogonalize a reduced representation against a basal expression profile.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param S Original input matrix (cells × genes).
    /// @param reduction_results Reduction field from reduceKernel(), in public
    ///                          Plan 02 layout {S_r, sigma, U, A, B}.
    /// @param basal_state Basal expression vector/matrix (genes × q).
    ///
    /// @return Corrected reduction field in the same public layout.
    template <typename T>
    arma::field<arma::mat> orthogonalizeBasal(T& S, arma::field<arma::mat>& reduction_results,
                                              arma::mat& basal_state);

    // ---- Operator-backed orthogonalization ------------------------------------------------

    /// @brief Orthogonalize a reduced representation against a batch design matrix
    ///        using a MatrixOperator for the backed matrix products.
    ///
    /// AnnData-native orientation (Plan 02): S operator is cells × genes.
    ///   Z = orth(S.rmatmat(design))  (genes × q via S' * design)
    ///   B = -(S.matmat(Z))           (cells × q via S * Z)
    ///
    /// @param S       Matrix operator (cells × genes, obs × var).
    /// @param reduction Current reduction state from reduceKernel(), in public
    ///                  Plan 02 layout {S_r, sigma, U, A, B}.
    /// @param design  Batch design matrix (cells × covariates).
    ///
    /// @return Updated reduction state in the same public layout.
    KernelReductionResult orthogonalizeBatchEffect_Operator(
        const MatrixOperator& S,
        const KernelReductionResult& reduction,
        const arma::mat& design);

    /// @brief Orthogonalize a reduced representation against a basal expression
    ///        profile using a MatrixOperator for the backed matrix products.
    ///
    /// AnnData-native orientation (Plan 02): S operator is cells × genes.
    ///   Z = orth(basal_state)   (genes × q)
    ///   B = -(S.matmat(Z))      (cells × q via S * Z)
    ///
    /// @param S           Matrix operator (cells × genes, obs × var).
    /// @param reduction   Current reduction state from reduceKernel(), in public
    ///                    Plan 02 layout {S_r, sigma, U, A, B}.
    /// @param basal_state Basal expression vector/matrix (genes × q).
    ///
    /// @return Updated reduction state in the same public layout.
    KernelReductionResult orthogonalizeBasal_Operator(
        const MatrixOperator& S,
        const KernelReductionResult& reduction,
        const arma::mat& basal_state);

} // namespace actionet

#endif //ACTIONET_ORTHOGONALIZATION_HPP
