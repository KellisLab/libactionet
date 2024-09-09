// Rcpp interface for `action` module
// Organized by module header in th order imported.
// [[Rcpp::interfaces(r, cpp)]]
#include <visualization/optimize_layout.hpp>
#include "actionet_r_config.h"

// aa ==================================================================================================================

// [[Rcpp::export]]
arma::mat run_uwot(arma::sp_mat& G, arma::mat& initial_position, UwotArgs uwot_args) {
    arma::mat embedding = optimize_layout_uwot(G, initial_position, std::move(uwot_args));
    return embedding;
}

