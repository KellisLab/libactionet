#include "utils_internal/utils_misc.hpp"

arma::mat one_hot_encoding(arma::vec V) {
    int n = V.n_elem;

    arma::vec vals = arma::unique(V);

    arma::uvec idx = arma::find(0 <= vals);
    vals = vals(idx);

    int k = vals.n_elem;
    arma::mat M = arma::zeros(n, k);
    for (int i = 0; i < k; i++) {
        arma::uvec idx = arma::find(V == vals(i));
        for (int j = 0; j < idx.n_elem; j++) {
            M(idx(j), i) = 1;
        }
    }

    return (M);
}

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