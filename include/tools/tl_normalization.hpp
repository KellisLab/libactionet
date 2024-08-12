#ifndef LIBACTIONET_TL_NORMALIZATION_HPP
#define LIBACTIONET_TL_NORMALIZATION_HPP

#include "libactionet_config.hpp"
#include "tools/tl_math.hpp"

arma::mat normalize_mat(arma::mat &X, int normalization, int dim);

arma::sp_mat normalize_mat(arma::sp_mat &X, int normalization, int dim);

arma::mat normalize_scores(arma::mat scores, int method = 1, int thread_no = 0);

// TODO: TF-IDF normalization (change name)
arma::sp_mat LSI(arma::sp_mat &S, double size_factor = 100000);

// TODO: Replace and remove with generic mat normalization functions
arma::sp_mat normalize_expression_profile(arma::sp_mat &S, int normalization = 1);

#endif //LIBACTIONET_TL_NORMALIZATION_HPP
