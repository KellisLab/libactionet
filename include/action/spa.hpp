// Successive projection algorithm (SPA)
#ifndef ACTIONET_SPA_HPP
#define ACTIONET_SPA_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {

    // To store the output of run_SPA()
    struct SPA_results {
        arma::uvec selected_columns;
        arma::vec column_norms;
    };

    // Solves separable NMF problem
    SPA_results run_SPA(arma::mat &A, int k);

} // namespace actionet

#endif //ACTIONET_SPA_HPP
