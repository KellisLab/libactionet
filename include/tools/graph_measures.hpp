// Tools for graph operations
#ifndef ACTIONET_GRAPH_MEASURES_HPP
#define ACTIONET_GRAPH_MEASURES_HPP

#include "libactionet_config.hpp"

namespace ACTIONet {

    arma::uvec compute_core_number(arma::sp_mat &G);

    arma::uvec compute_induced_core_number(arma::sp_mat &G, arma::uvec mask);

    arma::vec compute_archetype_core_centrality(arma::sp_mat &G, arma::uvec sample_assignments);

}  // namespace ACTIONet

#endif //ACTIONET_GRAPH_MEASURES_HPP
