#ifndef ACTIONET_MARKER_STATS_HPP
#define ACTIONET_MARKER_STATS_HPP

#include "libactionet_config.hpp"

// Forward declarations for backed operator types.
namespace actionet {
    class BackedSparseMatrixOperator;
    class BackedDenseMatrixOperator;
}

namespace actionet {
    /// @brief Compute marker statistics using network diffusion (legacy).
    ///
    /// @param G Graph adjacency matrix (cells x cells).
    /// @param S Feature matrix (cells x genes, i.e. obs x var).
    /// @param X Marker matrix (genes x labels).
    /// @param norm_method Graph normalization method.
    /// @param alpha Diffusion damping factor.
    /// @param max_it Maximum number of iterations.
    /// @param approx Use approximate diffusion.
    /// @param thread_no Number of threads (0 = auto).
    /// @param ignore_baseline Ignore baseline in scoring.
    ///
    /// @return Matrix of marker scores (cells x labels).
    arma::mat computeFeatureStats(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X, int norm_method = 2,
                                  double alpha = 0.85, int max_it = 5, bool approx = false, int thread_no = 0,
                                  bool ignore_baseline = false);

    /// @brief Compute marker statistics using VISION-style scoring.
    ///
    /// @param G Graph adjacency matrix (cells x cells).
    /// @param S Feature matrix (cells x genes, i.e. obs x var).
    /// @param X Marker matrix (genes x labels).
    /// @param norm_method Graph normalization method. (0 = pagerank, 2 = sym_pagerank).
    /// @param alpha Diffusion damping factor.
    /// @param max_it Maximum number of iterations.
    /// @param approx Use approximate diffusion.
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Matrix of marker scores (cells x labels).
    arma::mat computeFeatureStatsVision(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X, int norm_method = 2,
                                        double alpha = 0.85, int max_it = 5, bool approx = false, int thread_no = 0);

    /// @brief Backed sparse overload of computeFeatureStatsVision.
    ///
    /// Streams the expression matrix from disk via the backed operator.
    /// Output semantics are identical to the in-memory overload.
    arma::mat computeFeatureStatsVision(BackedSparseMatrixOperator& op,
                                        arma::sp_mat& G, arma::sp_mat& X,
                                        int norm_method = 2, double alpha = 0.85,
                                        int max_it = 5, bool approx = false,
                                        int thread_no = 0);

    /// @brief Backed dense overload of computeFeatureStatsVision.
    arma::mat computeFeatureStatsVision(BackedDenseMatrixOperator& op,
                                        arma::sp_mat& G, arma::sp_mat& X,
                                        int norm_method = 2, double alpha = 0.85,
                                        int max_it = 5, bool approx = false,
                                        int thread_no = 0);
}

#endif //ACTIONET_MARKER_STATS_HPP
