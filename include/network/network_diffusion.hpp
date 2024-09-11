// Network imputation using PageRank
#ifndef ACTIONET_NETWORK_DIFFUSION_HPP
#define ACTIONET_NETWORK_DIFFUSION_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // PageRank default
    // TODO: Unused, remove?
    arma::mat computeNetworkDiffusion(arma::sp_mat& G, arma::sp_mat& X0, double alpha = 0.85, int max_it = 3,
                                        int thread_no = 4);

    // PageRank (using cholmod)
    arma::mat computeNetworkDiffusionFast(arma::sp_mat& G, arma::sp_mat& X0, double alpha = 0.85, int max_it = 5,
                                             int thread_no = 0);

    // Fast approximate PageRank
    // norm_type: 0 (pagerank), 2 (sym_pagerank)
    arma::mat computeNetworkDiffusionApprox(arma::sp_mat& G, arma::mat& X, int norm_type = 0, double alpha = 0.85,
                                               int max_it = 5, double tol = 1E-8, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_NETWORK_DIFFUSION_HPP
