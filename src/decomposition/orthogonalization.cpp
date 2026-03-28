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
} // namespace

namespace actionet {

arma::field<arma::mat> deflateReduction(arma::field<arma::mat>& reduction_results,
                                        const arma::mat& A, const arma::mat& B) {
    stdout_printf("\tDeflating reduction ... ");
    FLUSH;

    if (reduction_results.n_elem != 5) {
        throw std::runtime_error("deflateReduction: expected reduction field layout {S_r, sigma, U, A, B}");
    }

    KernelReductionResult reduction = kernelResultFromField(reduction_results);
    SVDResult svd = svd_from_reduction_(reduction);

    PerturbedSVDResult prior_buf;
    const PerturbedSVDResult* prior_ptr = nullptr;
    if (has_prior_history_(reduction)) {
        prior_buf = prior_from_reduction_(reduction);
        prior_ptr = &prior_buf;
    }

    PerturbedSVDResult perturbed = deflate_reduction_struct_(svd, prior_ptr, A, B);
    stdout_printf("done\n");
    FLUSH;

    return kernelFieldFromResult(reduction_from_perturbed_(perturbed));
}

template <typename T>
arma::field<arma::mat> orthogonalizeBatchEffect(T& S, arma::field<arma::mat>& reduction_results, arma::mat& design) {
        stdout_printf("Orthogonalizing batch effect:\n");
        FLUSH;

        // S is cells × genes (Plan 02).  design is cells × q.
        // Z = S.t() * design: (cells × genes)' * (cells × q) = genes × q
        arma::mat Z = arma::mat(S.t() * design);
        gram_schmidt(Z);

        // B = -(S * Z): (cells × genes)(genes × q) = cells × q  — direct, no extra transpose
        arma::mat A = Z;
        arma::mat B = -arma::mat(S * Z);

        arma::field<arma::mat> perturbed_SVD = deflateReduction(reduction_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    template arma::field<arma::mat>
        orthogonalizeBatchEffect<arma::mat>(arma::mat& S, arma::field<arma::mat>& SVD_results, arma::mat& design);

    template arma::field<arma::mat>
        orthogonalizeBatchEffect<arma::sp_mat>(arma::sp_mat& S, arma::field<arma::mat>& SVD_results,
                                               arma::mat& design);

    template <typename T>
    arma::field<arma::mat> orthogonalizeBasal(T& S, arma::field<arma::mat>& reduction_results,
                                              arma::mat& basal_state) {
        stdout_printf("Orthogonalizing basal:\n");
        FLUSH;

        arma::mat Z = basal_state;
        gram_schmidt(Z);

        // S is cells × genes.  B = -(S * Z): (cells × genes)(genes × q) = cells × q
        arma::mat A = Z;
        arma::mat B = -arma::mat(S * Z);

        arma::field<arma::mat> perturbed_SVD = deflateReduction(reduction_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    template arma::field<arma::mat>
        orthogonalizeBasal<arma::mat>(arma::mat& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);

    template arma::field<arma::mat>
        orthogonalizeBasal<arma::sp_mat>(arma::sp_mat& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);

    // ---- Operator-backed orthogonalization ------------------------------------------------

    KernelReductionResult orthogonalizeBatchEffect_Operator(
        const MatrixOperator& S,
        const KernelReductionResult& reduction,
        const arma::mat& design) {

        stdout_printf("Orthogonalizing batch effect (operator):\n");
        FLUSH;

        SVDResult svd = svd_from_reduction_(reduction);
        PerturbedSVDResult prior_buf;
        const PerturbedSVDResult* prior_ptr = nullptr;
        if (has_prior_history_(reduction)) {
            prior_buf = prior_from_reduction_(reduction);
            prior_ptr = &prior_buf;
        }

        // S is cells × genes (obs × var).  design is (n_obs × q).
        // Z = S.t() * design = S' * design: rmatmat computes S' * X where X is (n_obs × q) → (n_var × q)
        // Old (genes × cells): S.matmat(design, Z) → (genes × cells)(cells × q) = genes × q
        // New (cells × genes): S.rmatmat(design, Z) → (cells × genes)'(cells × q) = genes × q
        arma::mat Z;
        S.rmatmat(design, Z);
        gram_schmidt(Z);

        // B = -(S * Z): matmat computes S * X where X is (n_var × q) → (n_obs × q) = cells × q
        // Old (genes × cells): S.rmatmat(Z, B_raw) → (genes × cells)'(genes × q) = cells × q
        // New (cells × genes): S.matmat(Z, B_raw)  → (cells × genes)(genes × q) = cells × q
        arma::mat B_raw;
        S.matmat(Z, B_raw);
        arma::mat B = -B_raw;

        stdout_printf("\tDeflating reduction ... ");
        FLUSH;
        PerturbedSVDResult result = deflate_reduction_struct_(svd, prior_ptr, Z, B);
        stdout_printf("done\n");
        FLUSH;

        return reduction_from_perturbed_(result);
    }

    KernelReductionResult orthogonalizeBasal_Operator(
        const MatrixOperator& S,
        const KernelReductionResult& reduction,
        const arma::mat& basal_state) {

        stdout_printf("Orthogonalizing basal (operator):\n");
        FLUSH;

        SVDResult svd = svd_from_reduction_(reduction);
        PerturbedSVDResult prior_buf;
        const PerturbedSVDResult* prior_ptr = nullptr;
        if (has_prior_history_(reduction)) {
            prior_buf = prior_from_reduction_(reduction);
            prior_ptr = &prior_buf;
        }

        arma::mat Z = basal_state;
        gram_schmidt(Z);

        // B = -(S * Z): matmat computes S * X → cells × q
        // Old (genes × cells): S.rmatmat(Z, B_raw) → (genes × cells)'(genes × q) = cells × q
        // New (cells × genes): S.matmat(Z, B_raw)  → (cells × genes)(genes × q) = cells × q
        arma::mat B_raw;
        S.matmat(Z, B_raw);
        arma::mat B = -B_raw;

        stdout_printf("\tDeflating reduction ... ");
        FLUSH;
        PerturbedSVDResult result = deflate_reduction_struct_(svd, prior_ptr, Z, B);
        stdout_printf("done\n");
        FLUSH;

        return reduction_from_perturbed_(result);
    }

} // namespace actionet
