// Convenience and backwards compatibility wrapper for "uwot_actionet"
// Abstracts out dependency of required argument classes and calls functions with arguments passed directly.
#ifndef ACTIONET_LAYOUT_NETWORK_HPP
#define ACTIONET_LAYOUT_NETWORK_HPP

#include "libactionet_config.hpp"
#include "visualization/OptimizerArgs.hpp"
#include "UwotArgs.hpp"

namespace actionet {
    /// @brief Compute a graph layout using uwot (convenience wrapper).
    ///
    /// @param G Graph adjacency matrix.
    /// @param initial_coordinates Initial coordinates (cells x dims).
    /// @param method Layout method ("umap", "tumap", "largevis").
    /// @param n_components Target embedding dimensions.
    /// @param spread UMAP spread parameter.
    /// @param min_dist UMAP min_dist parameter.
    /// @param n_epochs Number of epochs (0 = auto).
    /// @param learning_rate Learning rate.
    /// @param repulsion_strength Repulsion strength (gamma).
    /// @param negative_sample_rate Negative sampling rate.
    /// @param approx_pow Approximate power computation.
    /// @param pcg_rand Use PCG RNG.
    /// @param batch Use batch updates.
    /// @param grain_size Parallel grain size.
    /// @param seed Random seed.
    /// @param thread_no Number of threads (0 = auto).
    /// @param verbose Print progress messages.
    /// @param a UMAP a parameter (0 = auto).
    /// @param b UMAP b parameter (0 = auto).
    /// @param opt_method Optimizer method ("adam" or "sgd").
    /// @param alpha Optimizer alpha.
    /// @param beta1 Adam beta1.
    /// @param beta2 Adam beta2.
    /// @param eps Adam epsilon.
    ///
    /// @return Embedding coordinates (cells x n_components).
    arma::mat layoutNetwork(arma::sp_mat& G, arma::mat& initial_coordinates, std::string method = "umap",
                            unsigned int n_components = 2, float spread = 1, float min_dist = 1,
                            unsigned int n_epochs = 0, float learning_rate = LR_OPT_ALPHA, float repulsion_strength = 1,
                            float negative_sample_rate = 5, bool approx_pow = false, bool pcg_rand = true,
                            bool batch = true, unsigned int grain_size = 1, int seed = 0, int thread_no = 0,
                            bool verbose = true, float a = 0, float b = 0, std::string opt_method = "adam",
                            float alpha = LR_OPT_ALPHA, float beta1 = ADAM_BETA1, float beta2 = ADAM_BETA2,
                            float eps = ADAM_EPS);

    /// @brief Compute a graph layout using a pre-built argument struct.
    ///
    /// @param G Graph adjacency matrix.
    /// @param initial_coordinates Initial coordinates (cells x dims).
    /// @param uwot_args UMAP/t-UMAP configuration.
    /// @return Embedding coordinates (cells x n_components).
    arma::mat layoutNetwork(arma::sp_mat& G, arma::mat& initial_coordinates, UwotArgs uwot_args);
} // actionet

#endif //ACTIONET_LAYOUT_NETWORK_HPP
