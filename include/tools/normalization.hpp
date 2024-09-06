#ifndef ACTIONET_NORMALIZATION_HPP
#define ACTIONET_NORMALIZATION_HPP

#include "libactionet_config.hpp"

// TODO: are 0/1 needed? Redudant with arma::normalise().
// norm_type: 0 (column; pagerank), 1 (row), 2 (sym_pagerank)
arma::sp_mat normalize_adj(arma::sp_mat& G, int norm_type = 1);

namespace actionet {
    // `p` is p-norm. 1 ; 2: Euclidean norm. 0 (or negative) returns original X.
    template <typename T>
    T normalize_matrix(T& X, int p = 0, int dim = 0);

    arma::mat normalize_scores(arma::mat scores, int method = 1, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_NORMALIZATION_HPP
