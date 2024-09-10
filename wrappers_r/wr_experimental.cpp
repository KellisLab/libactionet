// Rcpp interface for `action` module
// Organized by module header in th order imported.
// [[Rcpp::interfaces(r, cpp)]]
#include <visualization/optimize_layout.hpp>
#include <visualization/color_map.hpp>
#include "actionet_r_config.h"

// aa ==================================================================================================================

// [[Rcpp::export]]
arma::mat run_uwot(arma::sp_mat& G, arma::mat& initial_position) {
    UwotArgs uwot_args;
    arma::mat embedding = actionet::optimize_layout_uwot(G, initial_position, uwot_args);
    return embedding;
}

// [[Rcpp::export]]
arma::mat computeNodeColors(arma::mat& coordinates, int thread_no) {
    arma::mat rgb = actionet::computeNodeColors(coordinates, thread_no);
    return rgb;
}



