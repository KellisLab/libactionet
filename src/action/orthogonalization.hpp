#ifndef ORTHOGONALIZATION_HPP
#define ORTHOGONALIZATION_HPP

#include "action.hpp"

// SVD algorithms
#define FULL_SVD -1
#define IRLB_ALG 0
#define HALKO_ALG 1
#define FENG_ALG 2

arma::field<arma::mat> deflate_reduction(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B);

#endif
