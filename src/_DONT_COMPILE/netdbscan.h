#ifndef LIBACTIONET_NETDBSCAN_H
#define LIBACTIONET_NETDBSCAN_H

#include "actionet.hpp"
#include "tools/tl_network.hpp"

// -1: Undefined, 0: Noise, 1...: Cluster IDs
#define UNDEFINED -1
#define NOISE 0

arma::vec NetDBSCAN(arma::sp_mat &G, int minPts, double eps = 0.5, double alpha_val = 0.85);

#endif //LIBACTIONET_NETDBSCAN_H
