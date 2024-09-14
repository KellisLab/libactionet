// Construct ACTIONet graph
#ifndef ACTIONET_BUILD_NETWORK_HPP
#define ACTIONET_BUILD_NETWORK_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Calls both buildNetwork_KstarNN() and buildNetwork_KNN()
    arma::sp_mat buildNetwork(const arma::mat& H, std::string algorithm = "k*nn", std::string distance_metric = "jsd",
                              double density = 1.0, int thread_no = 0, double M = 16, double ef_construction = 200,
                              double ef = 200, bool mutual_edges_only = true, int k = 10);
}

#endif //ACTIONET_BUILD_NETWORK_HPP
