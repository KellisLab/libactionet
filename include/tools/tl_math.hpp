#ifndef LIBACTIONET_TL_MATH_HPP
#define LIBACTIONET_TL_MATH_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_matrix.hpp"

arma::vec rank_vec(arma::vec x, int method = 0);

arma::mat zscore(arma::mat &A, int dim = 0, int thread_no = 1);

arma::mat robust_zscore(arma::mat &A, int dim = 0, int thread_no = 1);

arma::mat tzscoret(arma::mat &A);

arma::mat mean_center(arma::mat &A);

#endif //LIBACTIONET_TL_MATH_HPP
