#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include "config_arma.hpp"
#include "config_actionet.hpp"

// Exported functions
namespace ACTIONet {

    // generate_layout
    // Generate (U/t-U/PaC)MAP layout
    arma::field<arma::mat> layoutNetwork_xmap(arma::sp_mat &G, arma::mat &initial_position, bool presmooth_network,
                                              const std::string &method, double min_dist, double spread,
                                              double gamma, unsigned int n_epochs, int thread_no, int seed,
                                              double learning_rate, int sim2dist);

} // namespace ACTIONet

#endif
