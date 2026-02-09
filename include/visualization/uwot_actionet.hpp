// Wraps uwot 0.2.2 (https://github.com/jlmelville/uwot)
// This function and it's associated header tree implements the core graph optimization functionality of uwot for a
// precomputed graph and initial coordinates.
#ifndef ACTIONET_UWOT_ACTIONET_HPP
#define ACTIONET_UWOT_ACTIONET_HPP

#include "libactionet_config.hpp"
#include "UwotArgs.hpp"

// Implements `uwot::optimize_graph_layout()` for precomputed graph and initial coordinates
// Configured controlled via custom arguments structure `UwotArgs`
/// @brief Optimize a graph layout using uwot with custom arguments.
///
/// @param G Graph adjacency matrix.
/// @param initial_coordinates Initial coordinates (cells x dims).
/// @param uwot_args UMAP/t-UMAP configuration.
/// @return Embedding coordinates (cells x n_components).
arma::mat optimize_layout_uwot(arma::sp_mat& G, arma::mat& initial_coordinates, UwotArgs uwot_args);

#endif //ACTIONET_UWOT_ACTIONET_HPP
