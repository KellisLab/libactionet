// Singular value decomposition (SVD) algorithms
#include "decomposition/svd_main.hpp"
#include "decomposition/svd_irbla.hpp"
#include "decomposition/svd_feng.hpp"
#include "decomposition/svd_halko.hpp"
#include "utils_internal/utils_decomp.hpp"

// overloaded wrapper of arma::svd_econ to accept arma::mat and arma::sp_mat.
bool svd_econ_arma(arma::mat &U, arma::vec &s, arma::mat &V, const arma::sp_mat &S) {
    return arma::svd_econ(U, s, V, arma::mat(S));
}

bool svd_econ_arma(arma::mat &U, arma::vec &s, arma::mat &V, const arma::mat &S) {
    return arma::svd_econ(U, s, V, S);
}

namespace actionet {
    template<typename T>
    arma::field<arma::mat> runSVD(T &S, int k, int iter, int seed, int algorithm, int verbose) {

        char status_msg[100];
        snprintf(status_msg, 100, "Performing SVD input matrix using");
        stdout_printf(status_msg);

        arma::vec s;
        arma::mat U, V;
        arma::field<arma::mat> SVD_results(3);

        switch (algorithm) {
            case FULL_SVD:
                if (verbose) {
                    stdout_printf("%s %s", status_msg, "Armadillo");
                    FLUSH;
                };
                svd_econ_arma(U, s, V, S);
                SVD_results(0) = U;
                SVD_results(1) = s;
                SVD_results(2) = V;
                break;
            case IRLB_ALG:
                if (verbose) {
                    stdout_printf("%s %s", status_msg, "IRLBA");
                    FLUSH;
                };
                SVD_results = IRLB_SVD(S, k, iter, seed, verbose);
                break;
            case HALKO_ALG:
                if (verbose) {
                    stdout_printf("%s %s", status_msg, "Halko");
                    FLUSH;
                };
                SVD_results = HalkoSVD(S, k, iter, seed, verbose);
                break;
            case FENG_ALG:
                if (verbose) {
                    stdout_printf("%s %s", status_msg, "Fend");
                    FLUSH;
                };
                SVD_results = FengSVD(S, k, iter, seed, verbose);
                break;
            default:
                stderr_printf("Invalid SVD algorithm chosen (%d)\n", algorithm);
                stdout_printf("%s %s", status_msg, "Halko");
                FLUSH;
                SVD_results = IRLB_SVD(S, k, iter, seed, verbose);
                break;
        }

        return SVD_results;
    }

    template arma::field<arma::mat> runSVD<arma::mat>(arma::mat &S, int k, int iter, int seed, int algorithm,
                                                      int verbose);

    template arma::field<arma::mat> runSVD<arma::sp_mat>(arma::sp_mat &S, int k, int iter, int seed, int algorithm,
                                                         int verbose);

    arma::field<arma::mat> perturbedSVD(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B) {
        const arma::mat &U = SVD_results(0);
        const arma::mat &V = SVD_results(2);
        arma::vec s = SVD_results(1);

        int dim = U.n_cols;

        arma::vec s_prime;
        arma::mat U_prime, V_prime;

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

        arma::mat K1 = arma::zeros(s.n_elem + A.n_cols, s.n_elem + A.n_cols);
        for (int i = 0; i < s.n_elem; i++) {
            K1(i, i) = s(i);
        }

        arma::mat K2 = arma::join_vert(M, R_P) * arma::trans(arma::join_vert(N, R_Q));

        arma::mat K = K1 + K2;

        arma::svd(U_prime, s_prime, V_prime, K);

        arma::mat U_updated = arma::join_horiz(U, P) * U_prime;
        arma::mat V_updated = arma::join_horiz(V, Q) * V_prime;

        arma::field<arma::mat> output(5);
        output(0) = U_updated.cols(0, dim - 1);
        output(1) = s_prime(arma::span(0, dim - 1));
        output(2) = V_updated.cols(0, dim - 1);

        if ((SVD_results.n_elem == 5) && (SVD_results(3).n_elem != 0)) {
            output(3) = join_rows(SVD_results(3), A);
            output(4) = join_rows(SVD_results(4), B);
        } else {
            output(3) = A;
            output(4) = B;
        }

        return output;
    }
} // namespace actionet
