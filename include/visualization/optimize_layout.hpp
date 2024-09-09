#ifndef ACTIONET_OPTIMIZE_LAYOUT_HPP
#define ACTIONET_OPTIMIZE_LAYOUT_HPP
#include "libactionet_config.hpp"
#include "visualization/UwotArgs.hpp"

namespace actionet {
    arma::mat optimize_layout_uwot(arma::sp_mat& G, arma::mat& initial_position, UwotArgs uwot_args);
} // namespace actionet

#endif //ACTIONET_OPTIMIZE_LAYOUT_HPP
