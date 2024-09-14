// Tools for graph operations
#ifndef ACTIONET_NETWORK_MEASURES_HPP
#define ACTIONET_NETWORK_MEASURES_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    arma::uvec computeCoreness(arma::sp_mat& G);

    arma::vec computeArchetypeCentrality(arma::sp_mat& G, const arma::uvec& sample_assignments);
} // namespace actionet

#endif //ACTIONET_NETWORK_MEASURES_HPP
