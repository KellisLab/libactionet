// Singular value decomposition (SVD) algorithms
#include "decomposition/svd_main.hpp"
#include "decomposition/svd_irbla.hpp"
#include "decomposition/svd_feng.hpp"
#include "decomposition/svd_halko.hpp"
#include "utils_internal/utils_decomp.hpp"

namespace actionet {
    template <typename T>
    arma::field<arma::mat> runSVD(T& A, int k, int max_it, int seed, int algorithm, int verbose) {
        // out: U, sigma, V
        arma::field<arma::mat> out(3);

        // Default maximum iterations:
        if (max_it < 1) {
            switch (algorithm) {
                // Halko and Feng
                case 1:
                case 2:
                    max_it = 5;
                    break;
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
            case IRLB_ALG:
                out = IRLB_SVD(A, k, max_it, seed, verbose);
                break;
            case HALKO_ALG:
                out = HalkoSVD(A, k, max_it, seed, verbose);
                break;
            case FENG_ALG:
                out = FengSVD(A, k, max_it, seed, verbose);
                break;
            default:
                out = IRLB_SVD(A, k, max_it, seed, verbose);
                break;
        }

        return out;
    }

    template arma::field<arma::mat> runSVD<arma::mat>(arma::mat& A, int k, int max_it, int seed, int algorithm,
                                                      int verbose);
    template arma::field<arma::mat> runSVD<arma::sp_mat>(arma::sp_mat& A, int k, int max_it, int seed, int algorithm,
                                                         int verbose);

    arma::field<arma::mat> perturbedSVD(arma::field<arma::mat> SVD_results, arma::mat& A, arma::mat& B) {
        arma::field<arma::mat> out(5); // out: U', sigma', V', A, B

        const arma::mat& U = SVD_results(0);
        const arma::mat& V = SVD_results(2);
        arma::vec sigma = SVD_results(1);

        int dim = U.n_cols;

        arma::vec sigma_p;
        arma::mat U_p, V_p;

        arma::mat M = U.t() * A;
        arma::mat A_ortho_proj = A - U * M;
        arma::mat P = A_ortho_proj;
        gram_schmidt(P);
        arma::mat R_P = P.t() * A_ortho_proj;

        arma::mat N = V.t() * B;
        arma::mat B_ortho_proj = B - V * N;
        arma::mat Q = B_ortho_proj;
        gram_schmidt(Q);
        arma::mat R_Q = Q.t() * B_ortho_proj;

        arma::mat K1 = arma::zeros(sigma.n_elem + A.n_cols, sigma.n_elem + A.n_cols);
        for (int i = 0; i < sigma.n_elem; i++) {
            K1(i, i) = sigma(i);
        }

        arma::mat K2 = arma::join_vert(M, R_P) * arma::trans(arma::join_vert(N, R_Q));

        arma::mat K = K1 + K2;

        arma::svd(U_p, sigma_p, V_p, K);

        arma::mat U_updated = arma::join_horiz(U, P) * U_p;
        arma::mat V_updated = arma::join_horiz(V, Q) * V_p;

        // out: U_update, sigma', V_update, A, B
        out(0) = U_updated.cols(0, dim - 1);
        out(1) = sigma_p(arma::span(0, dim - 1));
        out(2) = V_updated.cols(0, dim - 1);

        if ((SVD_results.n_elem == 5) && (SVD_results(3).n_elem != 0)) {
            out(3) = join_rows(SVD_results(3), A);
            out(4) = join_rows(SVD_results(4), B);
        }
        else {
            out(3) = A;
            out(4) = B;
        }

        return out;
    }
} // namespace actionet
