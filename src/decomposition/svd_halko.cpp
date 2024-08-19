#include "decomposition/svd_halko.hpp"
#include "utils_internal/utils_decomp.hpp"

template<typename T>
arma::field<arma::mat> HalkoSVD(T &A, int dim, int iters, int seed, int verbose) {
    arma::field<arma::mat> results(3);

    int m = A.n_rows;
    int n = A.n_cols;

    dim = std::min(dim, std::min(m, n) - 2);

    int l = dim + 2;

    arma::vec s;
    arma::mat R, Q;
    arma::mat U, V, X;

    if (verbose) {
        stdout_printf("Halko (dense) -- A: %d x %d\n", (int) A.n_rows, (int) A.n_cols);
        FLUSH;
    }

    if (m < n) {
        R = randNorm(l, m, seed);
        Q = A.t() * R.t();
    } else {
        R = randNorm(n, l, seed);
        Q = A * R;
    }

    // Form a matrix Q whose columns constitute a well-conditioned basis for the
    // columns of the earlier Q.
    gram_schmidt(Q);

    if (m < n) {
        // Conduct normalized power iterations.
        for (int it = 1; it <= iters; it++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", it, iters);
                FLUSH;
            }

            Q = A * Q;
            gram_schmidt(Q);

            Q = A.t() * Q;
            gram_schmidt(Q);
        }

        X = arma::mat(A * Q);
        arma::svd_econ(U, s, V, X);
        V = Q * V;
    } else {
        // Conduct normalized power iterations.
        for (int it = 1; it <= iters; it++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", it, iters);
                FLUSH;
            }

            Q = A.t() * Q;
            gram_schmidt(Q);

            Q = A * Q; // Apply A to a random matrix, obtaining Q.
            gram_schmidt(Q);
        }

        // SVD Q' applied to the centered A to obtain approximations to the
        // singular values and right singular vectors of the A;
        X = arma::mat(Q.t() * A);
        arma::svd_econ(U, s, V, X);
        U = Q * U;
    }

    if (verbose) {
        stdout_printf("\r\tIteration %d/%d\n", iters, iters);
        FLUSH;
    }

    U.shed_cols(dim, dim + 1);
    s = s(arma::span(0, dim - 1));
    V.shed_cols(dim, dim + 1);

    results(0) = U;
    results(1) = s;
    results(2) = V;

    return (orient_SVD(results));
}

template arma::field<arma::mat> HalkoSVD<arma::mat>(arma::mat &A, int dim, int iters, int seed, int verbose);

template arma::field<arma::mat> HalkoSVD<arma::sp_mat>(arma::sp_mat &A, int dim, int iters, int seed, int verbose);