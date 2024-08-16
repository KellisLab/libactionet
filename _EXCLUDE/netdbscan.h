#ifndef ACTIONET_NETDBSCAN_H
#define ACTIONET_NETDBSCAN_H

#include "actionet.hpp"
#include "tools/graph_measures.hpp"

// -1: Undefined, 0: Noise, 1...: Cluster IDs
#define UNDEFINED -1
#define NOISE 0

arma::vec NetDBSCAN(arma::sp_mat &G, int minPts, double eps = 0.5, double alpha_val = 0.85);

#endif //ACTIONET_NETDBSCAN_H
