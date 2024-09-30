#ifndef ACTIONET_NORMALIZATION_HPP
#define ACTIONET_NORMALIZATION_HPP

#include "libactionet_config.hpp"

namespace actionet {
    // `p` is p-norm. 1: unit norm; 2: Euclidean norm. 0 (or negative) returns unnormalize matrix.
    template <typename T>
    T normalizeMatrix(T& X, unsigned int p = 1, unsigned int dim = 0);

    template <typename T>
    T scaleMatrix(T& X, arma::vec& v, unsigned int dim = 0);

    // norm_method: 0 (column; pagerank), 1 (row), 2 (sym_pagerank)
    arma::sp_mat normalizeGraph(arma::sp_mat& G, int norm_method = 1);


    arma::mat normalize_scores(arma::mat scores, int method = 1, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_NORMALIZATION_HPP
