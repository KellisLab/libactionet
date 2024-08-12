#include "tools/tl_math.hpp"

/* Rank a numeric vector giving ties their average rank */
arma::vec rank_vec(arma::vec x, int method) {
    int n = x.n_elem;
    arma::vec ranks(n);
    arma::uvec indx = arma::sort_index(x, "ascend");

    int ib = 0, i;
    double b = x[indx[0]];
    for (i = 1; i < n; ++i) {
        if (x[indx[i]] != b) { /* consecutive numbers differ */
            if (ib < i - 1) { /* average of sum of ranks */

                double rnk = method == 0 ? ((i - 1 + ib + 2) / 2.0) : (i - 1);
                for (int j = ib; j <= i - 1; ++j)
                    ranks[indx[j]] = rnk;
            } else {
                ranks[indx[ib]] = (double) (ib + 1);
            }
            b = x[indx[i]];
            ib = i;
        }
    }
    /* now check leftovers */
    if (ib == i - 1) /* last two were unique */
        ranks[indx[ib]] = (double) i;
    else { /* ended with ties */
        double rnk = method == 0 ? ((i - 1 + ib + 2) / 2.0) : (i - 1);
        for (int j = ib; j <= i - 1; ++j)
            ranks[indx[j]] = rnk;
    }

    return ranks;
}

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

arma::mat mean_center(arma::mat &A) {
    arma::mat A_centered = A;
    arma::rowvec mu = arma::rowvec(mean(A, 0));

    A_centered.each_row() -= mu;

    return A_centered;
}


