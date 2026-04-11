// Network imputation using PageRank
#ifndef ACTIONET_NETWORK_DIFFUSION_HPP
#define ACTIONET_NETWORK_DIFFUSION_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Diffuse scores over a graph using PageRank-style smoothing.
    ///
    /// @tparam T Dense matrix type for scores (cells x features).
    /// @param G Graph adjacency matrix.
    /// @param X0 Input scores (cells x features or vector).
    /// @param alpha Damping factor (0-1).
    /// @param max_it Maximum iterations.
    /// @param thread_no Number of threads (0 = auto).
    /// @param approx Use approximate diffusion. (true: chebyshev, false = power iteration).
    /// @param norm_method Normalization (0 = pagerank, 2 = sym_pagerank).
    /// @param tol Convergence tolerance.
    ///
    /// @return Diffused scores matrix.
    ///
    /// @note G is not modified.  Prior versions mutated G in-place
    ///       (normalize + scale); that is no longer the case.
    template <typename T>
    arma::mat computeNetworkDiffusion(const arma::sp_mat& G, T& X0, double alpha = 0.85, int max_it = 5,
                                      int thread_no = 0, bool approx = false, int norm_method = 0, double tol = 1E-8);
} // namespace actionet

#endif //ACTIONET_NETWORK_DIFFUSION_HPP
