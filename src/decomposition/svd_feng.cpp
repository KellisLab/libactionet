#include "decomposition/svd_feng.hpp"
#include "utils_internal/utils_decomp.hpp"

namespace actionet {
namespace {

// Compute SVD via eigendecomposition of A' * A.
arma::field<arma::mat> eigSVD(const arma::mat& A) {
    int n = A.n_cols;
    arma::mat B = arma::trans(A) * A;

    arma::vec d;
    arma::mat V;
    arma::eig_sym(d, V, B);
    d = sqrt(d);

    arma::sp_mat S(n, n);
    S.diag() = 1 / d;
    arma::mat U = (S * arma::trans(V)) * arma::trans(A);
    U = arma::trans(U);

    arma::field<arma::mat> out(3);
    out(0) = U;
    out(1) = d;
    out(2) = V;

    return out;
}

} // anonymous namespace

template <typename T>
arma::field<arma::mat> svdFeng(const T& A, int dim, int max_it, int seed, bool verbose) {
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
        Q = actionet::randNorm(n, dim + s, seed);
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
        Q = actionet::randNorm(m, dim + s, seed);
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

    actionet::orient_SVD(out);
    return out;
}

arma::field<arma::mat> svdFeng(const MatrixOperator& A, int dim, int max_it,
                               int seed, bool verbose) {
    const int s = 5;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (m < 2 || n < 2) {
        return arma::field<arma::mat>(3);
    }

    dim = std::min(dim, std::min(m, n) + s - 1);

    if (verbose) {
        stdout_printf("Feng (operator) -- A: %d x %d\n", m, n);
        FLUSH;
    }

    arma::vec sigma;
    arma::mat Q, L, U, V;
    arma::field<arma::mat> svd_out;

    if (m < n) {
        Q = actionet::randNorm(n, dim + s, seed); // n x (dim+s)
        arma::mat Aq;
        A.matmat(Q, Aq);                // m x (dim+s)
        Q = std::move(Aq);

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

            arma::mat AtQ;
            A.rmatmat(Q, AtQ);          // n x (dim+s)
            arma::mat AAtQ;
            A.matmat(AtQ, AAtQ);        // m x (dim+s)

            if (i == max_it) {
                svd_out = eigSVD(AAtQ);
                Q = svd_out(0);
            }
            else {
                arma::lu(L, U, AAtQ);
                Q = L;
            }
        }

        arma::mat AtQ_final;
        A.rmatmat(Q, AtQ_final);        // n x (dim+s)
        svd_out = eigSVD(AtQ_final);
        V = svd_out(0);
        sigma = arma::vec(svd_out(1));
        U = svd_out(2);

        U = Q * arma::fliplr(U.cols(s, dim + s - 1));
        V = arma::fliplr(V.cols(s, dim + s - 1));
        sigma = arma::flipud(sigma(arma::span(s, dim + s - 1)));
    }
    else {
        Q = actionet::randNorm(m, dim + s, seed); // m x (dim+s)
        arma::mat AtQ;
        A.rmatmat(Q, AtQ);              // n x (dim+s)
        Q = std::move(AtQ);

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

            arma::mat AQ;
            A.matmat(Q, AQ);            // m x (dim+s)
            arma::mat AtAQ;
            A.rmatmat(AQ, AtAQ);        // n x (dim+s)

            if (i == max_it) {
                svd_out = eigSVD(AtAQ);
                Q = svd_out(0);
            }
            else {
                arma::lu(L, U, AtAQ);
                Q = L;
            }
        }

        arma::mat AQ_final;
        A.matmat(Q, AQ_final);          // m x (dim+s)
        svd_out = eigSVD(AQ_final);
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
    actionet::orient_SVD(out);
    return out;
}

template arma::field<arma::mat> svdFeng<arma::mat>(const arma::mat& A, int dim, int max_it, int seed, bool verbose);

template arma::field<arma::mat> svdFeng<arma::sp_mat>(const arma::sp_mat& A, int dim, int max_it, int seed, bool verbose);

} // namespace actionet
