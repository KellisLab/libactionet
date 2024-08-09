#include "utils_misc.hpp"

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