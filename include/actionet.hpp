#ifndef ACTIONET_HPP
#define ACTIONET_HPP

#include "config_arma.hpp"
#include "config_interface.hpp"

namespace ACTIONet {
    // build_network
    // Construct network
    arma::sp_mat buildNetwork(arma::mat H, std::string algorithm = "k*nn", std::string distance_metric = "jsd",
                              double density = 1.0, int thread_no = 0, double M = 16, double ef_construction = 200,
                              double ef = 200, bool mutual_edges_only = true, int k = 10);

    // network_diffusion
    // PageRank default
    arma::mat compute_network_diffusion(arma::sp_mat &G, arma::sp_mat &X0, int thread_no = 4, double alpha = 0.85,
                                        int max_it = 3);

    // PageRank (using cholmod)
    arma::mat compute_network_diffusion_fast(arma::sp_mat &G, arma::sp_mat &X0, int thread_no = 4, double alpha = 0.85,
                                             int max_it = 5);

    // Fast approximate PageRank
    arma::mat compute_network_diffusion_Chebyshev(arma::sp_mat &P, arma::mat &X0, int thread_no, double alpha,
                                                  int max_it, double res_threshold);

} // namespace ACTIONet

#endif
