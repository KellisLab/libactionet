// Network imputation using PageRank
#ifndef LIBACTIONET_NETWORK_DIFFUSION_HPP
#define LIBACTIONET_NETWORK_DIFFUSION_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_matrix.hpp"
#include <cholmod.h>

// Exported
namespace ACTIONet {

    // PageRank default
    arma::mat compute_network_diffusion(arma::sp_mat &G, arma::sp_mat &X0, int thread_no = 4, double alpha = 0.85,
                                        int max_it = 3);

    // PageRank (using cholmod)
    arma::mat compute_network_diffusion_fast(arma::sp_mat &G, arma::sp_mat &X0, int thread_no = 4, double alpha = 0.85,
                                             int max_it = 5);

    // Fast approximate PageRank
    arma::mat compute_network_diffusion_Chebyshev(arma::sp_mat &P, arma::mat &X, int thread_no = 0, double alpha = 0.85,
                                                  int max_it = 5, double res_threshold = 1e-8);

}

#endif //LIBACTIONET_NETWORK_DIFFUSION_HPP
