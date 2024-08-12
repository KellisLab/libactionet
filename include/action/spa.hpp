// Successive projection algorithm (SPA)
#ifndef LIBACTIONET_SPA_HPP
#define LIBACTIONET_SPA_HPP

#include "libactionet_config.hpp"

// Exported
namespace ACTIONet {

    // To store the output of run_SPA()
    struct SPA_results {
        arma::uvec selected_columns;
        arma::vec column_norms;
    };

    // Solves separable NMF problem
    SPA_results run_SPA(arma::mat &A, int k);

} // namespace ACTIONet

#endif //LIBACTIONET_SPA_HPP
