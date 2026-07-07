// Kernel reduction interface for the ACTION algorithm.
//
// Provides both in-memory (dense/sparse) and operator-backed (OOM) entry
// points for computing the reduced ACTION kernel.  Result types are exported
// as typed structs; legacy arma::field wrappers are kept for backward
// compatibility with existing R and Python bindings.
//
// AnnData-native orientation contract (Plan 02):
//   S  : cells x genes  (obs x var)
//   S_r: cells x k
//   U  : genes x k      (gene loadings, unchanged)
//   A  : genes x p      (perturbation left,  unchanged)
//   B  : cells x p      (perturbation right, unchanged)
//
// The reduction pipeline is:
//   1. Compute (or accept precomputed) truncated SVD of S.
//   2. Compute perturbation matrices A (gene-space) and B (cell-space) from
//      column/row means of S (cells x genes).
//   3. Apply perturbedSVD with swapped A/B roles (B pertains to row-space = cells).
//   4. Round and scale the left singular vectors (U, cells x k) to produce S_r.

#ifndef ACTIONET_REDUCE_KERNEL_HPP
#define ACTIONET_REDUCE_KERNEL_HPP

#include "libactionet_config.hpp"
#include "decomposition/svd_main.hpp"

// Exported
namespace actionet {

    /// @brief Result of kernel reduction.
    ///
    /// AnnData-native orientation (cells x genes input contract):
    ///   - S_r: Reduced kernel matrix (cells × k), derived from the left singular
    ///          vectors of the perturbed decomposition, scaled by sigma and
    ///          discretised.  Was (k × cells) in the pre-Plan-02 contract.
    ///   - U  : Gene loadings (genes × k).  These correspond to the right singular
    ///          vectors of S (cells × genes).  Name kept as U for legacy compat.
    struct KernelReductionResult {
        arma::mat S_r;      ///< Reduced kernel (cells × k)    — was (k × cells)
        arma::vec sigma;    ///< Singular values (k)
        arma::mat U;        ///< Gene loadings / right singular vectors (genes × k)
        arma::mat A;        ///< Gene-space (col-space) perturbation (genes × p).
                            ///< Note: populated from the internal perturbedSVD's B term because
                            ///< S is cells × genes; kept named `A` for legacy field-layout compat.
        arma::mat B;        ///< Cell-space (row-space) perturbation (cells × p).
                            ///< Note: populated from the internal perturbedSVD's A term (see A above).
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

    /// @brief Apply ACTION kernel post-processing from a precomputed SVD and perturbation terms.
    ///
    /// S is treated as cells × genes.  perturbedSVD is called with B (cells × p) as
    /// the left (row-space) perturbation and A (genes × p) as the right (col-space)
    /// perturbation, matching the new orientation.  S_r is assembled from the left
    /// singular vectors (cells × k), not the right.
    ///
    /// @param svd  Precomputed truncated SVD of S (cells × genes).
    ///             svd.U is (cells × k), svd.V is (genes × k).
    /// @param A    Gene-space (col-space) perturbation (genes × p).
    /// @param B    Cell-space (row-space) perturbation (cells × p).
    KernelReductionResult applyKernelPostSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B);

    /// @brief Compute reduced ACTION kernel from precomputed SVD and operator-backed matrix.
    ///
    /// Computes perturbation terms via the operator, then delegates to applyKernelPostSVD.
    KernelReductionResult reduceKernelFromSVD_Operator(const MatrixOperator& S, const SVDResult& svd,
                                                       bool verbose = true);

    /// @brief Compute reduced ACTION kernel from operator-backed matrix (cells × genes) using selected SVD.
    ///
    /// @note Unavailable in R builds (PRIMME is Python-only in v1).
    KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, int k, int svd_alg = ALG_HALKO,
                                                int max_it = 0, int seed = 0, bool verbose = true);

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
    /// @param S      Input matrix (cells × genes, obs × var).
    /// @param svd    Precomputed truncated SVD of S.
    /// @param verbose Print status messages.
    template <typename T>
    KernelReductionResult reduceKernelFromSVD_InMemory(const T& S, const SVDResult& svd, bool verbose = true);

    // ---- Legacy in-memory entry point -----------------------------------------------------

    /// @brief Compute a reduced kernel matrix using truncated SVD.
    ///
    /// @tparam T Dense or sparse Armadillo matrix type.
    /// @param S Input matrix (<em>cells</em> × <em>genes</em>, i.e. obs × var).
    /// @param k Number of singular vectors to estimate.
    /// @param svd_alg SVD algorithm (see @c runSVD()).
    /// @param max_it Maximum number of SVD iterations.
    /// @param seed Random seed.
    /// @param verbose Print status messages.
    ///
    /// @return Field with 5 elements: {S_r (cells × k), sigma, U (genes × k), A, B}.
    template <typename T>
    arma::field<arma::mat> reduceKernel(T& S, int k, int svd_alg = 0, int max_it = 0,
                                        int seed = 0, bool verbose = true);
}

#endif //ACTIONET_REDUCE_KERNEL_HPP
