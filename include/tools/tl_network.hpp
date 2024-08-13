// Tools for graph operations
#ifndef LIBACTIONET_TL_NETWORK_HPP
#define LIBACTIONET_TL_NETWORK_HPP

#include "libactionet_config.hpp"

namespace ACTIONet {

    arma::uvec compute_core_number(arma::sp_mat &G);

    arma::vec compute_archetype_core_centrality(arma::sp_mat &G, arma::uvec sample_assignments);

}  // namespace ACTIONet

#endif //LIBACTIONET_TL_NETWORK_HPP
