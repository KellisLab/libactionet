// Miscellaneous internal helper functions
#ifndef ACTIONET_UTILS_MISC_HPP
#define ACTIONET_UTILS_MISC_HPP

#include "libactionet_config.hpp"

namespace actionet {

/// @brief Convert a label vector to one-hot encoding.
///
/// @param V Input label vector.
/// @return One-hot matrix.
arma::mat one_hot_encoding(const arma::vec& V);

/// @brief Rank a vector with optional method.
///
/// @param x Input vector.
/// @param method Ranking method code.
/// @return Ranked vector.
arma::vec rank_vec(arma::vec x, int method = 0);

} // namespace actionet

#endif //ACTIONET_UTILS_MISC_HPP
