#ifndef ACTIONET_UTILS_DECOMP_HPP
#define ACTIONET_UTILS_DECOMP_HPP

#include "libactionet_config.hpp"

namespace actionet {

/// @brief Gram-Schmidt orthogonalization.
void gram_schmidt(arma::mat &A);

/// @brief Generate standard normal random matrix.
arma::mat randNorm(int l, int m, int seed);

/// @brief Orient SVD components for sign consistency (modifies in-place).
void orient_SVD(arma::field<arma::mat>& SVD_res);

} // namespace actionet

#endif //ACTIONET_UTILS_DECOMP_HPP
