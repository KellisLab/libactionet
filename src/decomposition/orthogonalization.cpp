#include "decomposition/orthogonalization.hpp"
#include "decomposition/svd_main.hpp"
#include "utils_internal/utils_decomp.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
    actionet::SVDResult svd_from_reduction_(const actionet::KernelReductionResult& reduction) {
        if (reduction.S_r.n_cols != reduction.sigma.n_elem) {
            throw std::runtime_error("orthogonalization: S_r.n_cols != sigma.n_elem");
        }
        if (reduction.U.n_cols != reduction.sigma.n_elem) {
            throw std::runtime_error("orthogonalization: U.n_cols != sigma.n_elem");
        }
        if (reduction.A.n_elem != 0 && reduction.A.n_rows != reduction.U.n_rows) {
            throw std::runtime_error("orthogonalization: A.n_rows != U.n_rows (genes)");
        }
        if (reduction.B.n_elem != 0 && reduction.B.n_rows != reduction.S_r.n_rows) {
            throw std::runtime_error("orthogonalization: B.n_rows != S_r.n_rows (cells)");
        }
        if ((reduction.A.n_elem == 0) != (reduction.B.n_elem == 0)) {
            throw std::runtime_error("orthogonalization: A/B perturbation history must both be empty or both be present");
        }

        actionet::SVDResult svd;
        svd.U = reduction.S_r;
        svd.sigma = reduction.sigma;
        svd.V = reduction.U;

        const double zero_tol = std::numeric_limits<double>::epsilon();
        for (arma::uword i = 0; i < svd.sigma.n_elem; ++i) {
            const double sigma = svd.sigma(i);
            if (std::abs(sigma) <= zero_tol) {
                svd.U.col(i).zeros();
            } else {
                svd.U.col(i) /= sigma;
            }
        }

        return svd;
    }

    bool has_prior_history_(const actionet::KernelReductionResult& reduction) {
        return reduction.A.n_elem != 0 && reduction.B.n_elem != 0;
    }

    actionet::PerturbedSVDResult prior_from_reduction_(const actionet::KernelReductionResult& reduction) {
        actionet::PerturbedSVDResult prior;
        prior.A = reduction.B;  // row-space / cells
        prior.B = reduction.A;  // col-space / genes
        return prior;
    }

    actionet::KernelReductionResult reduction_from_perturbed_(const actionet::PerturbedSVDResult& perturbed) {
        actionet::KernelReductionResult out;
        out.sigma = perturbed.sigma;
        out.S_r = perturbed.U;
        for (arma::uword i = 0; i < out.S_r.n_cols; ++i) {
            out.S_r.col(i) *= out.sigma(i);
        }
        out.U = perturbed.V;
        out.A = perturbed.B;  // genes
        out.B = perturbed.A;  // cells
        return out;
    }

    // Threshold below which classical Gram-Schmidt is faster than LAPACK QR for
    // orthonormalising the (rows × c) batch / basal subspace matrix `Z`.  At
    // very small c the LAPACK dispatch overhead dominates the actual flops.
    // Mirrors the threshold used in perturbedSVD; see svd_main.cpp.
    static constexpr arma::uword ORTHOG_SMALL_C_THRESHOLD = 16;

    // Replace `Z` (rows × c) with an orthonormal basis for its column space.
    // Picks the cheaper of classical Gram-Schmidt (small c) and Householder QR
    // (large c).  Modifies `Z` in place.
    inline void orthonormalize_columns_(arma::mat& Z) {
        if (Z.n_cols == 0) return;
        if (Z.n_cols <= ORTHOG_SMALL_C_THRESHOLD) {
            actionet::gram_schmidt(Z);
        } else {
            arma::mat Q_orth, R_unused;
            arma::qr_econ(Q_orth, R_unused, Z);
            Z = std::move(Q_orth);
        }
    }

    actionet::PerturbedSVDResult deflate_reduction_struct_(const actionet::SVDResult& svd,
                                                           const actionet::PerturbedSVDResult* prior,
                                                           const arma::mat& A,
                                                           const arma::mat& B) {
        if (A.n_rows != svd.V.n_rows) {
            throw std::runtime_error("orthogonalization: gene-space perturbation rows must match reduction U rows (genes)");
        }
        if (B.n_rows != svd.U.n_rows) {
            throw std::runtime_error("orthogonalization: cell-space perturbation rows must match S_r rows (cells)");
        }
        if (A.n_cols != B.n_cols) {
            throw std::runtime_error("orthogonalization: gene-space and cell-space perturbation column counts must match");
        }

        arma::vec mu_A = arma::vec(arma::trans(arma::mean(A, 0)));
        arma::vec mu = B * mu_A;

        arma::mat A_aug = arma::join_rows(arma::ones(A.n_rows), A);   // genes x (q + 1)
        arma::mat B_aug = arma::join_rows(-mu, B);                    // cells x (q + 1)

        // perturbedSVD expects (left / row-space, right / col-space).
        return actionet::perturbedSVD(svd, B_aug, A_aug, prior);
    }

    // Deflate a public-layout reduction field {S_r, sigma, U, A, B} against perturbation
    // matrices `A` (genes x q) and `B` (cells x q).  Only used from the field-based
    // orthogonalize* overloads below.
    arma::field<arma::mat> deflate_reduction_field_(arma::field<arma::mat>& reduction_results,
                                                    const arma::mat& A, const arma::mat& B) {
        stdout_printf("\tDeflating reduction ... ");
        FLUSH;

        if (reduction_results.n_elem != 5) {
            throw std::runtime_error("deflateReduction: expected reduction field layout {S_r, sigma, U, A, B}");
        }

        actionet::KernelReductionResult reduction = actionet::kernelResultFromField(reduction_results);
        actionet::SVDResult svd = svd_from_reduction_(reduction);

        actionet::PerturbedSVDResult prior_buf;
        const actionet::PerturbedSVDResult* prior_ptr = nullptr;
        if (has_prior_history_(reduction)) {
            prior_buf = prior_from_reduction_(reduction);
            prior_ptr = &prior_buf;
        }

        actionet::PerturbedSVDResult perturbed = deflate_reduction_struct_(svd, prior_ptr, A, B);
        stdout_printf("done\n");
        FLUSH;

        return actionet::kernelFieldFromResult(reduction_from_perturbed_(perturbed));
    }
} // namespace

namespace actionet {

    namespace {
        // Shared body for the in-memory templated orthogonalize* variants.
        // The only difference between batch and basal is how the initial
        // subspace @c Z is constructed: batch uses @c S.t() * design, basal
        // uses @c basal_state directly. After orthonormalisation the flow
        // is identical (A = Z, B = -S*Z, deflate).
        template <typename T>
        arma::field<arma::mat> orthogonalize_common_(T& S,
                                                     arma::field<arma::mat>& reduction_results,
                                                     arma::mat&& Z_init,
                                                     const char* label) {
            stdout_printf("Orthogonalizing %s:\n", label);
            FLUSH;

            arma::mat Z = std::move(Z_init);
            orthonormalize_columns_(Z);

            // S is cells x genes.  B = -(S * Z): (cells x genes)(genes x q) = cells x q
            arma::mat A = Z;
            arma::mat B = arma::mat(S * Z);
            B *= -1.0;

            arma::field<arma::mat> perturbed_SVD = deflate_reduction_field_(reduction_results, A, B);
            FLUSH;
            return perturbed_SVD;
        }
    } // namespace

template <typename T>
arma::field<arma::mat> orthogonalizeBatchEffect(T& S, arma::field<arma::mat>& reduction_results, arma::mat& design) {
    // Z = S.t() * design: (cells x genes)' * (cells x q) = genes x q
    return orthogonalize_common_(S, reduction_results, arma::mat(S.t() * design), "batch effect");
}

    template arma::field<arma::mat>
        orthogonalizeBatchEffect<arma::mat>(arma::mat& S, arma::field<arma::mat>& SVD_results, arma::mat& design);

    template arma::field<arma::mat>
        orthogonalizeBatchEffect<arma::sp_mat>(arma::sp_mat& S, arma::field<arma::mat>& SVD_results,
                                               arma::mat& design);

    template <typename T>
    arma::field<arma::mat> orthogonalizeBasal(T& S, arma::field<arma::mat>& reduction_results,
                                              arma::mat& basal_state) {
        return orthogonalize_common_(S, reduction_results, arma::mat(basal_state), "basal");
    }

    template arma::field<arma::mat>
        orthogonalizeBasal<arma::mat>(arma::mat& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);

    template arma::field<arma::mat>
        orthogonalizeBasal<arma::sp_mat>(arma::sp_mat& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);

    // ---- Sparse one-hot fast path for batch correction ------------------------------------

    arma::field<arma::mat> orthogonalizeBatchEffect_sparse_labels(
        const arma::sp_mat& S,
        arma::field<arma::mat>& reduction_results,
        const arma::Col<arma::sword>& batch_labels,
        arma::uword n_batches) {

        stdout_printf("Orthogonalizing batch effect (sparse one-hot fast path):\n");
        FLUSH;

        const arma::uword n_cells = S.n_rows;
        const arma::uword n_genes = S.n_cols;
        if (batch_labels.n_elem != n_cells) {
            throw std::runtime_error(
                "orthogonalizeBatchEffect_sparse_labels: batch_labels length must equal n_cells");
        }
        if (n_batches == 0) {
            throw std::runtime_error(
                "orthogonalizeBatchEffect_sparse_labels: n_batches must be positive");
        }

        // Pass 1: build Z = S' * D  (genes × n_batches) directly from the CSC
        // nonzeros of S.  S is stored cells × genes, so iterating in column-major
        // order gives column = gene, row = cell.  Each nonzero v at (cell i, gene j)
        // contributes v to Z(j, label(i)).  Cells with negative labels are skipped.
        arma::mat Z(n_genes, n_batches, arma::fill::zeros);
        for (auto it = S.begin(); it != S.end(); ++it) {
            const arma::sword lbl = batch_labels(it.row());
            if (lbl < 0) continue;
            const arma::uword col = static_cast<arma::uword>(lbl);
            if (col >= n_batches) {
                throw std::runtime_error(
                    "orthogonalizeBatchEffect_sparse_labels: batch label out of range");
            }
            Z(it.col(), col) += (*it);
        }

        // Orthonormalize Z (genes × n_batches); GS for small c, QR for large.
        orthonormalize_columns_(Z);

        // B = -(S * Z): cells × n_batches.  Use the standard sparse-dense product;
        // this is a single pass over the nnz with width = Z.n_cols (after QR,
        // possibly truncated to rank ≤ n_batches).
        arma::sp_mat S_ref = S;  // make non-const for templated path
        arma::mat A = Z;
        arma::mat B = arma::mat(S_ref * Z);
        B *= -1.0;

        arma::field<arma::mat> perturbed_SVD = deflate_reduction_field_(reduction_results, A, B);
        FLUSH;
        return perturbed_SVD;
    }

    // ---- Operator-backed orthogonalization ------------------------------------------------

    namespace {
        // Shared body for operator-backed orthogonalization. Batch initialises
        // Z via @c S.rmatmat(design, Z); basal initialises Z directly from the
        // caller-supplied basal_state. The rest (orthonormalise, B = -S*Z,
        // deflate) is identical.
        KernelReductionResult orthogonalize_common_op_(const MatrixOperator& S,
                                                       const KernelReductionResult& reduction,
                                                       arma::mat&& Z_init,
                                                       const char* label) {
            stdout_printf("Orthogonalizing %s (operator):\n", label);
            FLUSH;

            SVDResult svd = svd_from_reduction_(reduction);
            PerturbedSVDResult prior_buf;
            const PerturbedSVDResult* prior_ptr = nullptr;
            if (has_prior_history_(reduction)) {
                prior_buf = prior_from_reduction_(reduction);
                prior_ptr = &prior_buf;
            }

            arma::mat Z = std::move(Z_init);
            orthonormalize_columns_(Z);

            // B = -(S * Z): matmat computes S * X -> cells x q
            arma::mat B;
            S.matmat(Z, B);
            B *= -1.0;

            stdout_printf("\tDeflating reduction ... ");
            FLUSH;
            PerturbedSVDResult result = deflate_reduction_struct_(svd, prior_ptr, Z, B);
            stdout_printf("done\n");
            FLUSH;

            return reduction_from_perturbed_(result);
        }
    } // namespace

    KernelReductionResult orthogonalizeBatchEffect_Operator(
        const MatrixOperator& S,
        const KernelReductionResult& reduction,
        const arma::mat& design) {

        // Z = S.t() * design: rmatmat computes S' * X where X is (n_obs x q) -> (n_var x q)
        arma::mat Z;
        S.rmatmat(design, Z);
        return orthogonalize_common_op_(S, reduction, std::move(Z), "batch effect");
    }

    KernelReductionResult orthogonalizeBasal_Operator(
        const MatrixOperator& S,
        const KernelReductionResult& reduction,
        const arma::mat& basal_state) {

        return orthogonalize_common_op_(S, reduction, arma::mat(basal_state), "basal");
    }

} // namespace actionet
