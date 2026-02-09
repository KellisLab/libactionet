#ifndef ACTIONET_MARKER_STATS_HPP
#define ACTIONET_MARKER_STATS_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Compute marker statistics using network diffusion (legacy).
    ///
    /// @param G Graph adjacency matrix.
    /// @param S Feature matrix (features x cells).
    /// @param X Marker matrix (features x labels).
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
    /// @param G Graph adjacency matrix.
    /// @param S Feature matrix (features x cells).
    /// @param X Marker matrix (features x labels).
    /// @param norm_method Graph normalization method. (0 = pagerank, 2 = sym_pagerank).
    /// @param alpha Diffusion damping factor.
    /// @param max_it Maximum number of iterations.
    /// @param approx Use approximate diffusion.
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Matrix of marker scores (cells x labels).
    arma::mat computeFeatureStatsVision(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X, int norm_method = 2,
                                        double alpha = 0.85, int max_it = 5, bool approx = false, int thread_no = 0);
}

#endif //ACTIONET_MARKER_STATS_HPP
