#ifndef ACTIONET_NORMALIZATION_HPP
#define ACTIONET_NORMALIZATION_HPP

#include "libactionet_config.hpp"

namespace actionet
{
    // `p` is p-norm. 1 ; 2: Euclidean norm. 0 (or negative) returns original X.
    template <typename T>
    T normalize_matrix(T& X, int p = 0, int dim = 0);

    arma::sp_mat normalize_adj(arma::sp_mat& G, int norm_type = 1);

    arma::mat normalize_scores(arma::mat scores, int method = 1, int thread_no = 0);

} // namespace actionet

#endif //ACTIONET_NORMALIZATION_HPP
