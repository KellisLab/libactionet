#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include "config_arma.hpp"
#include "libactionet_config.hpp"

// Exported functions
namespace ACTIONet {

    // generate_layout
    // Generate (U/t-U/PaC)MAP layout
    arma::field<arma::mat> layoutNetwork_xmap(arma::sp_mat &G, arma::mat &initial_position,
                                              bool presmooth_network = false, const std::string &method = "umap",
                                              double min_dist = 1, double spread = 1, double gamma = 1.0,
                                              unsigned int n_epochs = 500, int thread_no = 0, int seed = 0,
                                              double learning_rate = 1.0, int sim2dist = 2);

} // namespace ACTIONet

#endif
