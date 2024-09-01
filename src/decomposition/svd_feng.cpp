#include "decomposition/svd_feng.hpp"
#include "utils_internal/utils_matrix.hpp"
#include "utils_internal/utils_decomp.hpp"

template <typename T>
arma::field<arma::mat> FengSVD(T& A, int dim, int max_it, int seed, int verbose) {
    int s = 5;
    int m = A.n_rows;
    int n = A.n_cols;

    dim = std::min(dim, std::min(m, n) + s - 1);

    if (verbose) {
        stdout_printf("Feng -- A: %d x %d\n", (int)A.n_rows, (int)A.n_cols);
        FLUSH;
    }

    arma::vec sigma;
    arma::mat Q, L, U, V;
    arma::field<arma::mat> svd_out;

    if (m < n) {
        Q = randNorm(n, dim + s, seed);
        Q = A * Q;
        if (max_it == 0) {
            svd_out = eigSVD(Q);
            Q = svd_out(0);
        }
        else {
            arma::lu(L, U, Q);
            Q = L;
        }

        for (int i = 1; i <= max_it; i++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", i, max_it);
                FLUSH;
            }
            if (i == max_it) {
                svd_out = eigSVD(A * (arma::trans(A) * Q));
                Q = svd_out(0);
            }
            else {
                lu(L, U, A * (arma::trans(A) * Q));
                Q = L;
            }
        }

        svd_out = eigSVD(trans(A) * Q);
        V = svd_out(0);
        sigma = arma::vec(svd_out(1));
        U = svd_out(2);

        U = Q * arma::fliplr(U.cols(s, dim + s - 1));
        V = arma::fliplr(V.cols(s, dim + s - 1));
        sigma = arma::flipud(sigma(arma::span(s, dim + s - 1)));
    }
    else {
        Q = randNorm(m, dim + s, seed);
        Q = arma::trans(A) * Q;
        if (max_it == 0) {
            svd_out = eigSVD(Q);
            Q = svd_out(0);
        }
        else {
            arma::lu(L, U, Q);
            Q = L;
        }

        for (int i = 1; i <= max_it; i++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", i, max_it);
                FLUSH;
            }
            if (i == max_it) {
                svd_out = eigSVD(trans(A) * (A * Q));
                Q = svd_out(0);
            }
            else {
                arma::lu(L, U, arma::trans(A) * (A * Q));
                Q = L;
            }
        }

        svd_out = eigSVD(A * Q);
        U = svd_out(0);
        sigma = arma::vec(svd_out(1));
        V = svd_out(2);

        U = arma::fliplr(U.cols(s, dim + s - 1));
        V = Q * arma::fliplr(V.cols(s, dim + s - 1));
        sigma = arma::flipud(sigma(arma::span(s, dim + s - 1)));
    }

    if (verbose) {
        stdout_printf("\r\tIteration %d/%d\n", max_it, max_it);
        FLUSH;
    }


    arma::field<arma::mat> out(3); // out: U, sigma, V
    out(0) = U;
    out(1) = sigma;
    out(2) = V;

    return (orient_SVD(out));
}

template arma::field<arma::mat> FengSVD<arma::mat>(arma::mat& A, int dim, int max_it, int seed, int verbose);

template arma::field<arma::mat> FengSVD<arma::sp_mat>(arma::sp_mat& A, int dim, int max_it, int seed, int verbose);
