// Kernel reduction interface for the ACTION algorithm.
//
// Provides both in-memory (dense/sparse) and operator-backed (OOM) entry
// points for computing the reduced ACTION kernel.  Result types are exported
// as typed structs; legacy arma::field wrappers are kept for backward
// compatibility with existing R and Python bindings.
//
// The reduction pipeline is:
//   1. Compute (or accept precomputed) truncated SVD of S.
//   2. Compute perturbation matrices A, B from column/row means of S.
//   3. Apply perturbedSVD to the SVD + perturbation.
//   4. Round and scale the right singular vectors to produce S_r.

#ifndef ACTIONET_REDUCE_KERNEL_HPP
#define ACTIONET_REDUCE_KERNEL_HPP

#include "libactionet_config.hpp"
#include "decomposition/svd_main.hpp"

// Exported
namespace actionet {

    /// @brief Result of kernel reduction.
    ///
    /// Fields follow standard SVD naming conventions:
    ///   - U  : Left singular vectors of the perturbed decomposition (features × k).
    ///          These correspond to the "V" (gene loadings) output in some legacy
    ///          field-based interfaces.
    ///   - S_r: Reduced kernel matrix (k × cells), derived from the right singular
    ///          vectors scaled by sigma and discretised.
    struct KernelReductionResult {
        arma::mat S_r;      ///< Reduced kernel (k × cells)
        arma::vec sigma;    ///< Singular values (k)
        arma::mat U;        ///< Left singular vectors of perturbed SVD (features × k)
        arma::mat A;        ///< Accumulated left perturbation (features × p)
        arma::mat B;        ///< Accumulated right perturbation (cells × p)
    };

    // ---- Legacy field ↔ struct conversion helpers ----------------------------------------
    // Field layout: {S_r, sigma, U, A, B}  (5 elements)

    inline KernelReductionResult kernelResultFromField(const arma::field<arma::mat>& reduction) {
        KernelReductionResult out;
        out.S_r   = reduction(0);
        out.sigma = arma::vec(reduction(1));
        out.U     = reduction(2);
        out.A     = reduction(3);
        out.B     = reduction(4);
        return out;
    }

    inline arma::field<arma::mat> kernelFieldFromResult(const KernelReductionResult& reduction) {
        arma::field<arma::mat> out(5);
        out(0) = reduction.S_r;
        out(1) = reduction.sigma;
        out(2) = reduction.U;
        out(3) = reduction.A;
        out(4) = reduction.B;
        return out;
    }

    // ---- Operator-backed (OOM) entry points -----------------------------------------------

    /// @brief Compute perturbation matrices A and B from an operator-backed matrix.
    ///
    /// A and B encode centering corrections derived from row and column means of S.
    ///
    /// @param S  Matrix operator (features × cells).
    /// @param[out] A  Left perturbation matrix  (features × 2).
    /// @param[out] B  Right perturbation matrix (cells × 2).
    void computeKernelPerturbationTerms(const MatrixOperator& S, arma::mat& A, arma::mat& B);

    /// @brief Apply ACTION kernel post-processing from a precomputed SVD and perturbation terms.
    ///
    /// Applies perturbedSVD, then rounds and scales the right singular vectors to
    /// produce the reduced kernel S_r.
    ///
    /// @param svd  Precomputed truncated SVD of S.
    /// @param A    Left perturbation matrix  (features × p).
    /// @param B    Right perturbation matrix (cells × p).
    KernelReductionResult applyKernelPostSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B);

    /// @brief Compute reduced ACTION kernel from precomputed SVD and operator-backed matrix.
    ///
    /// Computes perturbation terms via the operator, then delegates to applyKernelPostSVD.
    KernelReductionResult reduceKernelFromSVD_Operator(const MatrixOperator& S, const SVDResult& svd,
                                                       bool verbose = true);

    /// @brief Compute reduced ACTION kernel from operator-backed matrix using PRIMME SVD.
    ///
    /// @note Unavailable in R builds (PRIMME is Python-only in v1).
    KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, int k, int max_it = 0,
                                                int seed = 0, bool verbose = true);

    // ---- In-memory precomputed SVD entry points -------------------------------------------

    /// @brief Compute reduced ACTION kernel from precomputed SVD and precomputed perturbation terms.
    ///
    /// Caller provides both the SVD and the perturbation matrices A/B.
    KernelReductionResult reduceKernelFromSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B);

    /// @brief Compute reduced ACTION kernel from precomputed SVD and an in-memory matrix.
    ///
    /// Computes perturbation terms directly from S (dense or sparse), then applies
    /// the post-SVD kernel pipeline.  This avoids the operator overhead for matrices
    /// that are already materialised in memory.
    ///
    /// @tparam T Dense (arma::mat) or sparse (arma::sp_mat) matrix type.
    /// @param S      Input matrix (features × cells).
    /// @param svd    Precomputed truncated SVD of S.
    /// @param verbose Print status messages.
    template <typename T>
    KernelReductionResult reduceKernelFromSVD_InMemory(const T& S, const SVDResult& svd, bool verbose = true);

    // ---- Legacy in-memory entry point -----------------------------------------------------

    /// @brief Compute a reduced kernel matrix using truncated SVD.
    ///
    /// @tparam T Dense or sparse Armadillo matrix type.
    /// @param S Input matrix (<em>features</em> × <em>cells</em>).
    /// @param k Number of singular vectors to estimate.
    /// @param svd_alg SVD algorithm (see @c runSVD()).
    /// @param max_it Maximum number of SVD iterations.
    /// @param seed Random seed.
    /// @param verbose Print status messages.
    ///
    /// @return Field with 5 elements: {S_r, sigma, U, A, B}.
    template <typename T>
    arma::field<arma::mat> reduceKernel(T& S, int k, int svd_alg = 0, int max_it = 0,
                                        int seed = 0, bool verbose = true);
}

#endif //ACTIONET_REDUCE_KERNEL_HPP
