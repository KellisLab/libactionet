// Miscellaneous internal helper functions
#ifndef ACTIONET_UTILS_MISC_HPP
#define ACTIONET_UTILS_MISC_HPP

#include "libactionet_config.hpp"

arma::mat one_hot_encoding(arma::vec V);

arma::vec rank_vec(arma::vec x, int method = 0);

#endif //ACTIONET_UTILS_MISC_HPP
