// Singular value decomposition (SVD) using Feng method
// From: Xu Feng, Yuyang Xie, and Yaohang Li, "Fast Randomzied SVD for Sparse
// Data," in Proc. the 10th Asian Conference on Machine Learning (ACML),
// Beijing, China, Nov. 2018.
#ifndef ACTIONET_SVD_FENG_HPP
#define ACTIONET_SVD_FENG_HPP

#include "libactionet_config.hpp"

arma::field<arma::mat> FengSVD(arma::sp_mat &A, int dim, int iters = 5, int seed = 0, int verbose = 1);

arma::field<arma::mat> FengSVD(arma::mat &A, int dim, int iters = 5, int seed = 0, int verbose = 1);

#endif //ACTIONET_SVD_FENG_HPP
