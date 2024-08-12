#ifndef LIBACTIONET_UTILS_GRAPH_H
#define LIBACTIONET_UTILS_GRAPH_H

#include "libactionet_config.hpp"

// TODO: Move to tools?
// Normalize adjacency matrix
arma::sp_mat normalize_adj(arma::sp_mat &G, int norm_type = 1);

#endif //LIBACTIONET_UTILS_GRAPH_H
