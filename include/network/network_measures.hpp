// Tools for graph operations
#ifndef ACTIONET_NETWORK_MEASURES_HPP
#define ACTIONET_NETWORK_MEASURES_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Compute k-core/coreness values for each node.
    ///
    /// @param G Graph adjacency matrix.
    /// @return Coreness per node.
    arma::uvec computeCoreness(arma::sp_mat& G);

    /// @brief Compute archetype-specific centrality scores.
    ///
    /// @param G Graph adjacency matrix.
    /// @param sample_assignments Assignment per node.
    /// @return Centrality per node.
    arma::vec computeArchetypeCentrality(arma::sp_mat& G, const arma::uvec& sample_assignments);
} // namespace actionet

#endif //ACTIONET_NETWORK_MEASURES_HPP
