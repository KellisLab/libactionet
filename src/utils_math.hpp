#ifndef UTILS_MATH_HPP
#define UTILS_MATH_HPP

#include "config_arma.hpp"

#include "utils_parallel.hpp"

arma::mat zscore(arma::mat &A, int dim, int thread_no = 0);

#endif
