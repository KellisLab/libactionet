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
    namespace {
        // Shared default max_it dispatcher for the algorithm switch used by
        // both runSVD (in-memory) and runSVD_Operator.
        int default_max_it(int algorithm) {
            switch (algorithm) {
                case ALG_HALKO:
                case ALG_FENG:
                    return 5;
#if !defined(LIBACTIONET_BUILD_R) || LIBACTIONET_BUILD_R == 0
                case ALG_PRIMME:
                    return 1000;
#endif
                case ALG_IRLB:
                default:
                    return 1000;
            }
        }
    } // anonymous namespace

    template <typename T>
    arma::field<arma::mat> runSVD(const T& A, int k, int max_it, int seed, const int algorithm, bool verbose) {
        // out: U, sigma, V
        arma::field<arma::mat> out(3);

        if (max_it < 1) {
            max_it = default_max_it(algorithm);
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

    template arma::field<arma::mat> runSVD<arma::mat>(const arma::mat& A, int k, int max_it, int seed, int algorithm,
                                                      bool verbose);
    template arma::field<arma::mat> runSVD<arma::sp_mat>(const arma::sp_mat& A, int k, int max_it, int seed, int algorithm,
                                                         bool verbose);

    SVDResult runSVD_Operator(const MatrixOperator& op, int k, int max_it, int seed, int algorithm, bool verbose) {
        if (max_it < 1) {
            max_it = default_max_it(algorithm);
        }

        switch (algorithm) {
            case ALG_HALKO:
                return svdResultFromField(svdHalko(op, k, max_it, seed, verbose));
            case ALG_FENG:
                return svdResultFromField(svdFeng(op, k, max_it, seed, verbose));
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

    // Threshold below which the optimized "split-GEMM + Householder QR" path is
    // slower than the legacy "join_horiz + classical Gram-Schmidt" path.  At
    // small `c` the LAPACK QR dispatch overhead exceeds the cost of an in-place
    // column-loop GS, and a single wide GEMM `[U|P] * U_p` is more cache-
    // friendly than two narrow GEMMs of widths k and c.  Empirically the
    // crossover sits near c ≈ k/2 ≈ 16 for k = 30.  Bench numbers in
    // docs/batch_correction_benchmark.md.
    static constexpr arma::uword PERTURBED_SVD_SMALL_C_THRESHOLD = 16;

    PerturbedSVDResult perturbedSVD(const SVDResult& svd,
                                    const arma::mat& Apert,
                                    const arma::mat& Bpert,
                                    const PerturbedSVDResult* prior) {
        const arma::mat& U = svd.U;
        const arma::mat& V = svd.V;
        const arma::vec& sigma = svd.sigma;

        const arma::uword k = U.n_cols;
        const arma::uword c = Apert.n_cols;
        const bool small_c = (c > 0) && (c <= PERTURBED_SVD_SMALL_C_THRESHOLD);

        // Project perturbation onto the existing SVD basis and compute orthogonal residuals.
        // M = U' * Apert  (k × c);  A_ortho = Apert - U * M  (m × c)
        arma::mat M = U.t() * Apert;
        arma::mat A_ortho = Apert - U * M;

        // Orthonormalise the residual column space.
        //   - Large c:  Householder QR (LAPACK, Level-3 BLAS) — `qr_econ`.
        //   - Small c:  Classical Gram-Schmidt (tight in-place column loop).
        // Both produce an orthonormal P; R_P is the upper-triangular part of
        // P' * A_ortho.  For the GS path we materialise R_P explicitly via a
        // narrow (c × c) GEMM, which is cheap when c is small.
        arma::mat P, R_P;
        if (small_c) {
            P = A_ortho;
            gram_schmidt(P);
            R_P = P.t() * A_ortho;          // c × c, exact when GS succeeded
        } else {
            arma::qr_econ(P, R_P, A_ortho);
        }

        arma::mat N = V.t() * Bpert;
        arma::mat B_ortho = Bpert - V * N;

        arma::mat Q, R_Q;
        if (small_c) {
            Q = B_ortho;
            gram_schmidt(Q);
            R_Q = Q.t() * B_ortho;
        } else {
            arma::qr_econ(Q, R_Q, B_ortho);
        }

        // Assemble the (k + c) × (k + c) inner matrix K block-wise without join_vert/trans:
        //   K = [[Σ + M N',   M R_Q'],
        //        [R_P N',     R_P R_Q']]
        const arma::uword K_dim = k + c;
        arma::mat K(K_dim, K_dim, arma::fill::none);
        // Top-left block: Σ + M*N'
        if (k > 0) {
            arma::mat top_left = M * N.t();
            top_left.diag() += sigma;
            K.submat(0, 0, k - 1, k - 1) = top_left;
        }
        if (c > 0) {
            // Top-right: M*R_Q'  (k × c)
            if (k > 0) {
                K.submat(0, k, k - 1, K_dim - 1) = M * R_Q.t();
                // Bottom-left: R_P*N'  (c × k)
                K.submat(k, 0, K_dim - 1, k - 1) = R_P * N.t();
            }
            // Bottom-right: R_P*R_Q'  (c × c)
            K.submat(k, k, K_dim - 1, K_dim - 1) = R_P * R_Q.t();
        }

        arma::vec sigma_p;
        arma::mat U_p, V_p;
        arma::svd(U_p, sigma_p, V_p, K);

        // Expand back to feature/sample space.
        //   - Large c:  Two narrow GEMMs `U·U_p_top + P·U_p_bot` (avoids the
        //               (m × (k+c)) join_horiz allocation and the wasted
        //               work on c discarded columns).
        //   - Small c:  Single wide GEMM `[U|P] · U_p[:, :k]` is more
        //               cache-friendly when c << k and the join_horiz cost
        //               is small relative to the GEMM.
        PerturbedSVDResult out;
        if (k == 0) {
            out.U.set_size(U.n_rows, 0);
            out.V.set_size(V.n_rows, 0);
            out.sigma.set_size(0);
        } else {
            arma::mat U_p_k = U_p.cols(0, k - 1);
            arma::mat V_p_k = V_p.cols(0, k - 1);
            out.sigma = sigma_p.subvec(0, k - 1);
            if (c == 0) {
                out.U = U * U_p_k;
                out.V = V * V_p_k;
            } else if (small_c) {
                out.U = arma::join_horiz(U, P) * U_p_k;
                out.V = arma::join_horiz(V, Q) * V_p_k;
            } else {
                out.U = U * U_p_k.head_rows(k) + P * U_p_k.tail_rows(c);
                out.V = V * V_p_k.head_rows(k) + Q * V_p_k.tail_rows(c);
            }
        }

        // Accumulate perturbation history via insert_cols (avoids constructing a fresh
        // (rows × p_total) matrix on every call when used cumulatively).
        if (prior != nullptr && prior->A.n_elem != 0) {
            out.A = prior->A;
            out.A.insert_cols(out.A.n_cols, Apert);
            out.B = prior->B;
            out.B.insert_cols(out.B.n_cols, Bpert);
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
