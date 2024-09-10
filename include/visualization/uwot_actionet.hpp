#ifndef ACTIONET_UWOT_ACTIONET_HPP
#define ACTIONET_UWOT_ACTIONET_HPP

#include "libactionet_config.hpp"
#include "UwotArgs.hpp"

namespace actionet {
    arma::mat optimize_layout_uwot(arma::sp_mat& G, arma::mat& initial_position, UwotArgs uwot_args);
}

#endif //ACTIONET_UWOT_ACTIONET_HPP
