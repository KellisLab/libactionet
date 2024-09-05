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

// Exported functions
namespace actionet {
    // Generate (U/t-U/PaC)MAP layout
    arma::field<arma::mat> layoutNetwork_xmap(arma::sp_mat& G, arma::mat& initial_position,
                                              const std::string& method = "umap", double min_dist = 1,
                                              double spread = 1, double gamma = 1.0, unsigned int n_epochs = 500,
                                              double learning_rate = 1.0, int seed = 0, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_GENERATE_LAYOUT_HPP
