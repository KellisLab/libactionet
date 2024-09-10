#ifndef ACTIONET_COLOR_MAP_HPP
#define ACTIONET_COLOR_MAP_HPP

#include "libactionet_config.hpp"

namespace actionet {
    arma::mat computeNodeColors(const arma::mat& coordinates, int thread_no);
} // actionet

#endif //ACTIONET_COLOR_MAP_HPP
