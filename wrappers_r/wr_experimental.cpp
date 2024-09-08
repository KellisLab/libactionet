// Rcpp interface for `action` module
// Organized by module header in th order imported.
// [[Rcpp::interfaces(r, cpp)]]
// #include <visualization/generate_layout_umappp.hpp>

#include <visualization/generate_layout_umappp.hpp>
#include "actionet_r_config.h"

// generate_layout ==================================================================================================================


// [[Rcpp::export]]
arma::mat layoutNetwork_umappp(arma::sp_mat& G, arma::mat& initial_embedding, int thread_no = 0) {

    arma::mat out = actionet::layoutNetwork_umappp(G, initial_embedding, thread_no);

    return out;

}

