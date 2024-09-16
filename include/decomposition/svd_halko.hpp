// Singular value decomposition (SVD) with Halko method
// Implemented from: N Halko, P. G Martinsson, and J. A Tropp. Finding structure with
// randomness: Probabilistic algorithms for constructing approximate matrix
// decompositions. Siam Review, 53(2):217-288, 2011.
#ifndef ACTIONET_SVD_HALKO_HPP
#define ACTIONET_SVD_HALKO_HPP

#include "libactionet_config.hpp"

template <typename T>
arma::field<arma::mat> svdHalko(T& A, int dim, int iters = 5, int seed = 0, bool verbose = true);

#endif //ACTIONET_SVD_HALKO_HPP
