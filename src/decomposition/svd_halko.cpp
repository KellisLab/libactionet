#include "decomposition/svd_halko.hpp"
#include "utils_internal/utils_decomp.hpp"

template <typename T>
arma::field<arma::mat> svdHalko(const T& A, int dim, int iters, int seed, bool verbose) {
    arma::field<arma::mat> out(3); // out: U, sigma, V

    arma::uword m = A.n_rows;
    arma::uword n = A.n_cols;

    const arma::uword mn_min = std::min(m, n);
    if (static_cast<arma::uword>(dim) + 2 > mn_min)
        dim = static_cast<int>(mn_min) - 2;
    if (dim < 1) dim = 1;

    int l = dim + 2;

    arma::vec sigma;
    arma::mat R, Q;
    arma::mat U, V, X;

    if (verbose) {
        stdout_printf("Halko -- A: %d x %d\n", (int)A.n_rows, (int)A.n_cols);
        FLUSH;
    }

    if (m < n) {
        R = actionet::randNorm(l, static_cast<int>(m), seed);
        Q = A.t() * R.t();
    }
    else {
        R = actionet::randNorm(static_cast<int>(n), l, seed);
        Q = A * R;
    }

    // Form a matrix Q whose columns constitute a well-conditioned basis for the
    // columns of the earlier Q.
    actionet::gram_schmidt(Q);

    if (m < n) {
        // Conduct normalized power iterations.
        for (int it = 1; it <= iters; it++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", it, iters);
                FLUSH;
            }

            Q = A * Q;
            actionet::gram_schmidt(Q);

            Q = A.t() * Q;
            actionet::gram_schmidt(Q);
        }

        X = arma::mat(A * Q);
        arma::svd_econ(U, sigma, V, X);
        V = Q * V;
    }
    else {
        // Conduct normalized power iterations.
        for (int it = 1; it <= iters; it++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", it, iters);
                FLUSH;
            }

            Q = A.t() * Q;
            actionet::gram_schmidt(Q);

            Q = A * Q; // Apply A to a random matrix, obtaining Q.
            actionet::gram_schmidt(Q);
        }

        // SVD Q' applied to the centered A to obtain approximations to the
        // singular values and right singular vectors of the A;
        X = arma::mat(Q.t() * A);
        arma::svd_econ(U, sigma, V, X);
        U = Q * U;
    }

    if (verbose) {
        stdout_printf("\r\tIteration %d/%d\n", iters, iters);
        FLUSH;
    }

    U.shed_cols(dim, dim + 1);
    sigma = sigma(arma::span(0, dim - 1));
    V.shed_cols(dim, dim + 1);

    out(0) = U;
    out(1) = sigma;
    out(2) = V;

    actionet::orient_SVD(out);
    return out;
}

arma::field<arma::mat> svdHalko(const actionet::MatrixOperator& A, int dim, int iters,
                                int seed, bool verbose) {
    arma::field<arma::mat> out(3); // out: U, sigma, V

    const arma::uword rows_uw = A.rows();
    const arma::uword cols_uw = A.cols();
    if (rows_uw < 2 || cols_uw < 2) {
        return out;
    }
    const int m = static_cast<int>(rows_uw);
    const int n = static_cast<int>(cols_uw);

    dim = std::min(dim, std::min(m, n) - 2);
    if (dim < 1) {
        dim = 1;
    }
    const int l = dim + 2;

    arma::vec sigma;
    arma::mat R, Q;
    arma::mat U, V, X;

    if (verbose) {
        stdout_printf("Halko (operator) -- A: %d x %d\n", m, n);
        FLUSH;
    }

    if (m < n) {
        R = actionet::randNorm(l, m, seed);       // l x m
        A.rmatmat(R.t(), Q);            // n x l
    }
    else {
        R = actionet::randNorm(n, l, seed);       // n x l
        A.matmat(R, Q);                 // m x l
    }

    actionet::gram_schmidt(Q);

    if (m < n) {
        for (int it = 1; it <= iters; it++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", it, iters);
                FLUSH;
            }

            arma::mat tmp_m;
            A.matmat(Q, tmp_m);         // m x l
            Q = std::move(tmp_m);
            actionet::gram_schmidt(Q);

            arma::mat tmp_n;
            A.rmatmat(Q, tmp_n);        // n x l
            Q = std::move(tmp_n);
            actionet::gram_schmidt(Q);
        }

        A.matmat(Q, X);                 // m x l
        arma::svd_econ(U, sigma, V, X);
        V = Q * V;
    }
    else {
        for (int it = 1; it <= iters; it++) {
            if (verbose) {
                stderr_printf("\r\tIteration %d/%d", it, iters);
                FLUSH;
            }

            arma::mat tmp_n;
            A.rmatmat(Q, tmp_n);        // n x l
            Q = std::move(tmp_n);
            actionet::gram_schmidt(Q);

            arma::mat tmp_m;
            A.matmat(Q, tmp_m);         // m x l
            Q = std::move(tmp_m);
            actionet::gram_schmidt(Q);
        }

        arma::mat Xt;
        A.rmatmat(Q, Xt);               // n x l  == A'Q
        X = Xt.t();                     // l x n  == Q'A
        arma::svd_econ(U, sigma, V, X);
        U = Q * U;
    }

    if (verbose) {
        stdout_printf("\r\tIteration %d/%d\n", iters, iters);
        FLUSH;
    }

    U.shed_cols(dim, dim + 1);
    sigma = sigma(arma::span(0, dim - 1));
    V.shed_cols(dim, dim + 1);

    out(0) = U;
    out(1) = sigma;
    out(2) = V;
    actionet::orient_SVD(out);
    return out;
}

template arma::field<arma::mat> svdHalko<arma::mat>(const arma::mat& A, int dim, int iters, int seed, bool verbose);

template arma::field<arma::mat> svdHalko<arma::sp_mat>(const arma::sp_mat& A, int dim, int iters, int seed, bool verbose);
