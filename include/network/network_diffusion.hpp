// Network imputation using PageRank
#ifndef ACTIONET_NETWORK_DIFFUSION_HPP
#define ACTIONET_NETWORK_DIFFUSION_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // approx: false (standard [fast] PageRank), true (Chebyshev PageRank)
    // norm_type: 0 (pagerank), 2 (sym_pagerank)
    template <typename T>
    arma::mat computeNetworkDiffusion(arma::sp_mat& G, T& X0, double alpha = 0.85, int max_it = 5,
                                      int thread_no = 0, bool approx = false, int norm_type = 0, double tol = 1E-8);
} // namespace actionet

#endif //ACTIONET_NETWORK_DIFFUSION_HPP
