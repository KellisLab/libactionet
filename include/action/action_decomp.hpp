// Main ACTION decomposition
#ifndef ACTIONET_ACTION_DECOMP_HPP
#define ACTIONET_ACTION_DECOMP_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet
{
    // To store the output of run_ACTION()
    struct ACTION_results
    {
        arma::field<arma::uvec> selected_cols;
        arma::field<arma::mat> H;
        arma::field<arma::mat> C;
    };

    // Runs ACTION decomposition
    ACTION_results
    run_ACTION(arma::mat& S_r, int k_min, int k_max, int normalization = 0, int max_it = 100, double min_delta = 1e-6,
               int thread_no = 0);
} // namespace actionet


#endif //ACTIONET_ACTION_DECOMP_HPP
