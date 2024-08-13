#ifndef LIBACTIONET_UTILS_STATS_HPP
#define LIBACTIONET_UTILS_STATS_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_matrix.hpp"

arma::mat zscore(arma::mat &A, int dim = 0, int thread_no = 1);

arma::mat robust_zscore(arma::mat &A, int dim = 0, int thread_no = 1);

// TODO: Unused. Remove?
arma::mat tzscoret(arma::mat &A);

arma::mat mean_center(arma::mat &A);

#endif //LIBACTIONET_UTILS_STATS_HPP
