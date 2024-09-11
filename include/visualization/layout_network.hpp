// Convenience and backwards compatibility wrapper for "uwot_actionet"
// Abstracts out dependency of required argument classes and calls functions with arguments passed directly.
#ifndef ACTIONET_LAYOUT_NETWORK_HPP
#define ACTIONET_LAYOUT_NETWORK_HPP

#include "libactionet.hpp"
#include "UwotArgs.hpp"

namespace actionet {
    arma::mat layoutNetwork(arma::sp_mat& G, arma::mat& initial_coordinates, std::string method = "umap",
                            unsigned int n_components = 2, float spread = 1, float min_dist = 1,
                            unsigned int n_epochs = 0, float learning_rate = LR_OPT_ALPHA, float repulsion_strength = 1,
                            float negative_sample_rate = 5, bool approx_pow = false, bool pcg_rand = true,
                            bool batch = true, unsigned int grain_size = 1, int seed = 0, int thread_no = 0,
                            bool verbose = true, float a = 0, float b = 0, std::string opt_method = "adam",
                            float alpha = LR_OPT_ALPHA, float beta1 = ADAM_BETA1, float beta2 = ADAM_BETA2,
                            float eps = ADAM_EPS);

    arma::mat layoutNetwork(arma::sp_mat& G, arma::mat& initial_coordinates, UwotArgs uwot_args);
} // actionet

#endif //ACTIONET_LAYOUT_NETWORK_HPP
