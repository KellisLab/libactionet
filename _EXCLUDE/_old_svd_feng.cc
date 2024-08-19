#include "decomposition/svd_feng.hpp"
#include "utils_internal/utils_matrix.hpp"
#include "utils_internal/utils_decomp.hpp"

arma::field<arma::mat> FengSVD(arma::sp_mat &A, int dim, int iters, int seed, int verbose) {
    int s = 5;
    int m = A.n_rows;
    int n = A.n_cols;

    dim = std::min(dim, std::min(m, n) + s - 1);

    if (verbose) {
        stdout_printf("Feng (sparse) -- A: %d x %d\n", (int) A.n_rows, (int) A.n_cols);
        FLUSH;
    }

    arma::vec S;
    arma::mat Q, L, U, V;
    arma::field<arma::mat> SVD_out;

    if (m < n) {
        randNorm(n, dim + s, seed);
        Q = A * Q;
        if (iters == 0) {
            SVD_out = eigSVD(Q);
            Q = SVD_out(0);
        } else {
            lu(L, U, Q);
            Q = L;
        }

        for (int i = 1; i <= iters; i++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", i, iters);
                FLUSH;
            }
            if (i == iters) {
                SVD_out = eigSVD(A * (trans(A) * Q));
                Q = SVD_out(0);
            } else {
                lu(L, U, A * (trans(A) * Q));
                Q = L;
            }
        }

        SVD_out = eigSVD(trans(A) * Q);
        V = SVD_out(0);
        S = arma::vec(SVD_out(1));
        U = SVD_out(2);

        U = Q * arma::fliplr(U.cols(s, dim + s - 1));
        V = arma::fliplr(V.cols(s, dim + s - 1));
        S = arma::flipud(S(arma::span(s, dim + s - 1)));
    } else {
        Q = randNorm(m, dim + s, seed);
        Q = arma::trans(A) * Q;
        if (iters == 0) {
            SVD_out = eigSVD(Q);
            Q = SVD_out(0);
        } else {
            lu(L, U, Q);
            Q = L;
        }

        for (int i = 1; i <= iters; i++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", i, iters);
                FLUSH;
            }

            if (i == iters) {
                SVD_out = eigSVD(trans(A) * (A * Q));
                Q = SVD_out(0);
            } else {
                arma::lu(L, U, arma::trans(A) * (A * Q));
                Q = L;
            }
        }

        SVD_out = eigSVD(A * Q);
        U = SVD_out(0);
        S = arma::vec(SVD_out(1));
        V = SVD_out(2);

        U = arma::fliplr(U.cols(s, dim + s - 1));
        V = Q * arma::fliplr(V.cols(s, dim + s - 1));
        S = arma::flipud(S(arma::span(s, dim + s - 1)));
    }

    if (verbose) {
        stdout_printf("\r\tIteration %d/%d\n", iters, iters);
        FLUSH;
    }

    arma::field<arma::mat> out(3);
    out(0) = U;
    out(1) = S;
    out(2) = V;

    return (orient_SVD(out));
}

arma::field<arma::mat> FengSVD(arma::mat &A, int dim, int iters, int seed, int verbose) {
    int s = 5;
    int m = A.n_rows;
    int n = A.n_cols;

    dim = std::min(dim, std::min(m, n) + s - 1);

    if (verbose) {
        stdout_printf("Feng (dense) -- A: %d x %d\n", (int) A.n_rows, (int) A.n_cols);
        FLUSH;
    }

    arma::vec S;
    arma::mat Q, L, U, V;
    arma::field<arma::mat> SVD_out;

    if (m < n) {
        Q = randNorm(n, dim + s, seed);

        Q = A * Q;
        if (iters == 0) {
            SVD_out = eigSVD(Q);
            Q = SVD_out(0);
        } else {
            arma::lu(L, U, Q);
            Q = L;
        }

        for (int i = 1; i <= iters; i++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", i, iters);
                FLUSH;
            }
            if (i == iters) {
                SVD_out = eigSVD(A * (trans(A) * Q));
                Q = SVD_out(0);
            } else {
                lu(L, U, A * (trans(A) * Q));
                Q = L;
            }
        }

        SVD_out = eigSVD(trans(A) * Q);
        V = SVD_out(0);
        S = arma::vec(SVD_out(1));
        U = SVD_out(2);

        U = Q * arma::fliplr(U.cols(s, dim + s - 1));
        V = arma::fliplr(V.cols(s, dim + s - 1));
        S = arma::flipud(S(arma::span(s, dim + s - 1)));
    } else {
        Q = randNorm(m, dim + s, seed);
        Q = arma::trans(A) * Q;
        if (iters == 0) {
            SVD_out = eigSVD(Q);
            Q = SVD_out(0);
        } else {
            arma::lu(L, U, Q);
            Q = L;
        }

        for (int i = 1; i <= iters; i++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", i, iters);
                FLUSH;
            }
            if (i == iters) {
                SVD_out = eigSVD(trans(A) * (A * Q));
                Q = SVD_out(0);
            } else {
                arma::lu(L, U, arma::trans(A) * (A * Q));
                Q = L;
            }
        }

        SVD_out = eigSVD(A * Q);
        U = SVD_out(0);
        S = arma::vec(SVD_out(1));
        V = SVD_out(2);

        U = arma::fliplr(U.cols(s, dim + s - 1));
        V = Q * arma::fliplr(V.cols(s, dim + s - 1));
        S = arma::flipud(S(arma::span(s, dim + s - 1)));
    }

    if (verbose) {
        stdout_printf("\r\tIteration %d/%d\n", iters, iters);
        FLUSH;
    }

    arma::field<arma::mat> out(3);
    out(0) = U;
    out(1) = S;
    out(2) = V;

    return (orient_SVD(out));
}
