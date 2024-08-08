#ifndef ACTIONET_HPP
#define ACTIONET_HPP

#include "config_arma.hpp"
#include "config_actionet.hpp"

namespace ACTIONet {
    // build_network
    // Construct network
    arma::sp_mat buildNetwork(arma::mat H, std::string algorithm = "k*nn", std::string distance_metric = "jsd",
                              double density = 1.0, int thread_no = 0, double M = 16, double ef_construction = 200,
                              double ef = 200, bool mutual_edges_only = true, int k = 10);

} // namespace ACTIONet

#endif
