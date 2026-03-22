// Construct ACTIONet graph
#ifndef ACTIONET_BUILD_NETWORK_HPP
#define ACTIONET_BUILD_NETWORK_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Build a cell-cell network from archetype weights.
    ///
    /// @param H Archetype weights (cells x k, AnnData-native orientation).
    /// @param algorithm "k*nn" or "knn".
    /// @param distance_metric "jsd", "l2", or "ip".
    /// @param density Graph density factor.
    /// @param thread_no Number of threads (0 = auto).
    /// @param M HNSW parameter M.
    /// @param ef_construction HNSW construction depth.
    /// @param ef HNSW search depth.
    /// @param mutual_edges_only Keep only mutual neighbors.
    /// @param k Number of neighbors for knn.
    ///
    /// @return Sparse adjacency matrix (cells x cells).
    arma::sp_mat buildNetwork(const arma::mat& H, std::string algorithm = "k*nn", std::string distance_metric = "jsd",
                              double density = 1.0, int thread_no = 0, double M = 16, double ef_construction = 200,
                              double ef = 200, bool mutual_edges_only = true, int k = 10);
}

#endif //ACTIONET_BUILD_NETWORK_HPP
