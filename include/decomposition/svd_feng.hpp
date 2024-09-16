// Singular value decomposition (SVD) using Feng method
// From: Xu Feng, Yuyang Xie, and Yaohang Li, "Fast Randomzied SVD for Sparse
// Data," in Proc. the 10th Asian Conference on Machine Learning (ACML),
// Beijing, China, Nov. 2018.
#ifndef ACTIONET_SVD_FENG_HPP
#define ACTIONET_SVD_FENG_HPP

#include "libactionet_config.hpp"

template <typename T>
arma::field<arma::mat> svdFeng(T& A, int dim, int max_it = 5, int seed = 0, bool verbose = true);

#endif //ACTIONET_SVD_FENG_HPP
