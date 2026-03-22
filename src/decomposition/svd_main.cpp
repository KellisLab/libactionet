// Singular value decomposition (SVD) algorithms
#include "decomposition/svd_main.hpp"
#include "decomposition/svd_irbla.hpp"
#include "decomposition/svd_feng.hpp"
#include "decomposition/svd_halko.hpp"
#if !defined(LIBACTIONET_BUILD_R) || LIBACTIONET_BUILD_R == 0
#include "decomposition/svd_primme.hpp"
#endif
#include "utils_internal/utils_decomp.hpp"
#include <stdexcept>

namespace actionet {
    template <typename T>
    arma::field<arma::mat> runSVD(T& A, int k, int max_it, int seed, const int algorithm, bool verbose) {
        // out: U, sigma, V
        arma::field<arma::mat> out(3);

        // Default maximum iterations:
        if (max_it < 1) {
            switch (algorithm) {
                // Halko and Feng
                case ALG_HALKO:
                case ALG_FENG:
                    max_it = 5;
                    break;
#if !defined(LIBACTIONET_BUILD_R) || LIBACTIONET_BUILD_R == 0
                case ALG_PRIMME:
                    max_it = 1000;
                    break;
#endif
                default:
                    // IRLB
                    max_it = 1000;
                    break;
            }
        }

        if (verbose) {
            stdout_printf("Performing SVD using ");
        }

        switch (algorithm) {
            case ALG_HALKO:
                out = svdHalko(A, k, max_it, seed, verbose);
                break;
            case ALG_FENG:
                out = svdFeng(A, k, max_it, seed, verbose);
                break;
#if !defined(LIBACTIONET_BUILD_R) || LIBACTIONET_BUILD_R == 0
            case ALG_PRIMME:
                out = svdPRIMME(A, k, max_it, seed, verbose);
                break;
#endif
            case ALG_IRLB:
            default:
                out = svdIRLB(A, k, max_it, seed, verbose);
        }

        return out;
    }

    template arma::field<arma::mat> runSVD<arma::mat>(arma::mat& A, int k, int max_it, int seed, int algorithm,
                                                      bool verbose);
    template arma::field<arma::mat> runSVD<arma::sp_mat>(arma::sp_mat& A, int k, int max_it, int seed, int algorithm,
                                                         bool verbose);

    SVDResult runSVD_Halko_Operator(const MatrixOperator& op, int k, int iters, int seed, bool verbose) {
        arma::field<arma::mat> out = svdHalko(op, k, iters, seed, verbose);
        return svdResultFromField(out);
    }

    SVDResult runSVD_Feng_Operator(const MatrixOperator& op, int k, int max_it, int seed, bool verbose) {
        arma::field<arma::mat> out = svdFeng(op, k, max_it, seed, verbose);
        return svdResultFromField(out);
    }

    SVDResult runSVD_Operator(const MatrixOperator& op, int k, int max_it, int seed, int algorithm, bool verbose) {
        if (max_it < 1) {
            switch (algorithm) {
                case ALG_HALKO:
                case ALG_FENG:
                    max_it = 5;
                    break;
#if !defined(LIBACTIONET_BUILD_R) || LIBACTIONET_BUILD_R == 0
                case ALG_PRIMME:
                    max_it = 1000;
                    break;
#endif
                default:
                    max_it = 1000;
                    break;
            }
        }

        switch (algorithm) {
            case ALG_HALKO:
                return runSVD_Halko_Operator(op, k, max_it, seed, verbose);
            case ALG_FENG:
                return runSVD_Feng_Operator(op, k, max_it, seed, verbose);
#if !defined(LIBACTIONET_BUILD_R) || LIBACTIONET_BUILD_R == 0
            case ALG_PRIMME:
                return runSVD_PRIMME_Operator(op, k, max_it, seed, verbose);
#endif
            case ALG_IRLB:
            default: {
                arma::field<arma::mat> result = svdIRLB(op, k, max_it, seed, verbose);
                return svdResultFromField(result);
            }
        }
    }

    // ---- Struct-based perturbedSVD (primary implementation) --------------------------------
    //
    // Implements the Brand (2006) rank-preserving SVD update.  Given A ≈ U Σ V' and additive
    // perturbation Apert * Bpert', this computes the updated truncated SVD while preserving the
    // original rank (number of kept singular values).
    //
    // When @p prior is non-null and carries non-empty perturbation terms, the new Apert/Bpert
    // are horizontally concatenated onto the prior's A/B for cumulative tracking.

    PerturbedSVDResult perturbedSVD(const SVDResult& svd,
                                    const arma::mat& Apert,
                                    const arma::mat& Bpert,
                                    const PerturbedSVDResult* prior) {
        const arma::mat& U = svd.U;
        const arma::mat& V = svd.V;
        const arma::vec& sigma = svd.sigma;

        int dim = static_cast<int>(U.n_cols);

        // Project perturbation onto the existing SVD basis and compute orthogonal residuals.
        arma::mat M = U.t() * Apert;
        arma::mat A_ortho_proj = Apert - U * M;
        arma::mat P = A_ortho_proj;
        gram_schmidt(P);
        arma::mat R_P = P.t() * A_ortho_proj;

        arma::mat N = V.t() * Bpert;
        arma::mat B_ortho_proj = Bpert - V * N;
        arma::mat Q = B_ortho_proj;
        gram_schmidt(Q);
        arma::mat R_Q = Q.t() * B_ortho_proj;

        // Build the inner (dim + p) × (dim + p) matrix and compute its full SVD.
        arma::mat K1 = arma::zeros(sigma.n_elem + Apert.n_cols, sigma.n_elem + Apert.n_cols);
        for (arma::uword i = 0; i < sigma.n_elem; i++) {
            K1(i, i) = sigma(i);
        }

        arma::mat K2 = arma::join_vert(M, R_P) * arma::trans(arma::join_vert(N, R_Q));
        arma::mat K = K1 + K2;

        arma::vec sigma_p;
        arma::mat U_p, V_p;
        arma::svd(U_p, sigma_p, V_p, K);

        arma::mat U_updated = arma::join_horiz(U, P) * U_p;
        arma::mat V_updated = arma::join_horiz(V, Q) * V_p;

        PerturbedSVDResult out;
        out.U     = U_updated.cols(0, dim - 1);
        out.sigma = sigma_p(arma::span(0, dim - 1));
        out.V     = V_updated.cols(0, dim - 1);

        // Accumulate perturbation history.
        if (prior != nullptr && prior->A.n_elem != 0) {
            out.A = arma::join_rows(prior->A, Apert);
            out.B = arma::join_rows(prior->B, Bpert);
        } else {
            out.A = Apert;
            out.B = Bpert;
        }

        return out;
    }

    // ---- Legacy field-based perturbedSVD (thin wrapper) ------------------------------------

    arma::field<arma::mat> perturbedSVD(const arma::field<arma::mat>& SVD_results,
                                        const arma::mat& A, const arma::mat& B) {
        SVDResult svd;
        svd.U     = SVD_results(0);
        svd.sigma = arma::vec(SVD_results(1));
        svd.V     = SVD_results(2);

        // If the field carries 5 elements with non-empty prior perturbation terms,
        // build a PerturbedSVDResult to represent them.
        PerturbedSVDResult prior_buf;
        const PerturbedSVDResult* prior_ptr = nullptr;
        if (SVD_results.n_elem == 5 && SVD_results(3).n_elem != 0) {
            prior_buf.A = SVD_results(3);
            prior_buf.B = SVD_results(4);
            prior_ptr = &prior_buf;
        }

        PerturbedSVDResult result = perturbedSVD(svd, A, B, prior_ptr);

        arma::field<arma::mat> out(5);
        out(0) = result.U;
        out(1) = result.sigma;
        out(2) = result.V;
        out(3) = result.A;
        out(4) = result.B;
        return out;
    }

} // namespace actionet
