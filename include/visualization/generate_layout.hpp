#ifndef ACTIONET_GENERATE_LAYOUT_HPP
#define ACTIONET_GENERATE_LAYOUT_HPP

#include "libactionet_config.hpp"

// Exported functions
namespace actionet {
    // Generate (U/t-U/PaC)MAP layout
    arma::field<arma::mat> layoutNetwork_xmap(arma::sp_mat& G, arma::mat& initial_position,
                                              const std::string& method = "umap", double min_dist = 1,
                                              double spread = 1, double gamma = 1.0, unsigned int n_epochs = 500,
                                              double learning_rate = 1.0, int seed = 0, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_GENERATE_LAYOUT_HPP
