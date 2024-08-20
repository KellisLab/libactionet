#ifndef ACTIONET_NORMALIZATION_HPP
#define ACTIONET_NORMALIZATION_HPP

#include "libactionet_config.hpp"

namespace actionet
{
    // p = 2 is Euclidean norm. 0 (or negative )returns original X.
    template <typename T>
    T normalize_mat(T& X, int p = 0, int dim = 0);

    // arma::mat normalize_mat(const arma::mat &X, int p = 0, int dim = 0);

    // arma::sp_mat normalize_mat(const arma::sp_mat &X, int normalization, int dim = 0);

    arma::sp_mat normalize_adj(arma::sp_mat& G, int norm_type = 1);

    arma::mat normalize_scores(arma::mat scores, int method = 1, int thread_no = 0);

    // TODO: TF-IDF normalization (change name)
    arma::sp_mat LSI(const arma::sp_mat& S, double size_factor = 100000);
} // namespace actionet

#endif //ACTIONET_NORMALIZATION_HPP
