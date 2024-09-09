#ifndef ACTIONET_OPTIMIZE_LAYOUT_HPP
#define ACTIONET_OPTIMIZE_LAYOUT_HPP
#include "libactionet_config.hpp"
#include "visualization/UwotArgs.hpp"

arma::mat optimize_layout_uwot(arma::sp_mat& G, arma::mat& initial_position, UwotArgs uwot_args);

#endif //ACTIONET_OPTIMIZE_LAYOUT_HPP

