#include "utils_internal/utils_stats.hpp"
#include "utils_internal/utils_parallel.hpp"

arma::mat zscore(arma::mat &A, int dim, int thread_no) {
    int N = A.n_cols;
    if (dim != 0) {
        N = A.n_rows;
    }

    mini_thread::parallelFor(
            0, N,
            [&](size_t j) {
                arma::vec v = A.col(j);
                if (dim == 0) {
                    v = A.col(j);
                } else {
                    v = A.row(j);
                }
                double mu = arma::mean(v);
                double sigma = arma::stddev(v);

                arma::vec z = (v - mu) / sigma;
                if (dim == 0) {
                    A.col(j) = z;
                } else {
                    A.row(j) = z;
                }
            },
            thread_no);
    A.replace(arma::datum::nan, 0); // replace each NaN with 0

    return A;
}

arma::mat robust_zscore(arma::mat &A, int dim, int thread_no) {
    int N = A.n_cols;
    if (dim != 0) {
        N = A.n_rows;
    }

    mini_thread::parallelFor(
            0, N,
            [&](size_t j) {
                arma::vec v = A.col(j);
                if (dim == 0) {
                    v = A.col(j);
                } else {
                    v = A.row(j);
                }
                double med = arma::median(v);
                double mad = arma::median(arma::abs(v - med));

                arma::vec z = (v - med) / mad;
                if (dim == 0) {
                    A.col(j) = z;
                } else {
                    A.row(j) = z;
                }
            },
            thread_no);
    A.replace(arma::datum::nan, 0); // replace each NaN with 0

    return A;
}

arma::mat tzscoret(arma::mat &A) {
    arma::mat At = A.t();
    A = zscore(At);
    return (A.t());
}

arma::mat mean_center(const arma::mat &A) {
    arma::mat A_centered = A;
    arma::rowvec mu = arma::rowvec(mean(A, 0));

    A_centered.each_row() -= mu;

    return A_centered;
}

