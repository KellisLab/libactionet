#ifndef UTILS_GRAPH_H
#define UTILS_GRAPH_H

#include "config_arma.hpp"

// Normalize adjacency matrix
arma::sp_mat normalize_adj(arma::sp_mat &G, int norm_type = 1);

#endif //UTILS_GRAPH_H
