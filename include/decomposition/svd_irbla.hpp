// Singular value decomposition (SVD) using IRLBA
// Implemented from irlba R package (https://github.com/bwlewis/irlba)
#ifndef ACTIONET_SVD_IRBLA_HPP
#define ACTIONET_SVD_IRBLA_HPP

#include "libactionet_config.hpp"

arma::field<arma::mat> svdIRLB(arma::sp_mat& A, int dim, int iters = 1000, int seed = 0, int verbose = 1);

arma::field<arma::mat> svdIRLB(arma::mat& A, int dim, int iters = 1000, int seed = 0, int verbose = 1);

#endif //ACTIONET_SVD_IRBLA_HPP
