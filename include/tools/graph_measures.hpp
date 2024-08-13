// Tools for graph operations
#ifndef LIBACTIONET_GRAPH_MEASURES_HPP
#define LIBACTIONET_GRAPH_MEASURES_HPP

#include "libactionet_config.hpp"

namespace ACTIONet {

    arma::uvec compute_core_number(arma::sp_mat &G);

    arma::uvec compute_induced_core_number(arma::sp_mat &G, arma::uvec mask);

    arma::vec compute_archetype_core_centrality(arma::sp_mat &G, arma::uvec sample_assignments);

}  // namespace ACTIONet

#endif //LIBACTIONET_GRAPH_MEASURES_HPP
