#ifndef ACTIONET_GENERATE_LAYOUT_HPP
#define ACTIONET_GENERATE_LAYOUT_HPP

#include "libactionet_config.hpp"

// Constants
#define NEGATIVE_SAMPLE_RATE 3.0
#define UMAP_SEED 0
#define GAMMA 1.0
#define ADAM_ALPHA 1.0 /*same as learning_rate*/
#define ADAM_BETA1 0.5 /*only adam: between 0 and 1*/
#define ADAM_BETA2 0.9 /*only adam: between 0 and 1*/
#define ADAM_EPS 1e-7  /*only adam: between 1e-8 and 1e-3*/

// FUnctions: internal
arma::sp_mat smoothKNN(arma::sp_mat &D, int max_iter = 64, double epsilon = 1e-6, double bandwidth = 1.0,
                       double local_connectivity = 1.0, double min_k_dist_scale = 1e-3, double min_sim = 1e-8,
                       int thread_no = 0);

// Exported functions
namespace actionet {

    // Generate (U/t-U/PaC)MAP layout
    arma::field<arma::mat> layoutNetwork_xmap(arma::sp_mat &G, arma::mat &initial_position,
                                              bool presmooth_network = false, const std::string &method = "umap",
                                              double min_dist = 1, double spread = 1, double gamma = 1.0,
                                              unsigned int n_epochs = 500, int thread_no = 0, int seed = 0,
                                              double learning_rate = 1.0, int sim2dist = 2);

} // namespace actionet

#endif //ACTIONET_GENERATE_LAYOUT_HPP
