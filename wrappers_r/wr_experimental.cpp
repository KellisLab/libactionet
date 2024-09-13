// Rcpp interface for `action` module
// Organized by module header in th order imported.
// [[Rcpp::interfaces(r, cpp)]]

#include "actionet_r_config.h"

// experimental ==================================================================================================================

// [[Rcpp::export]]
arma::mat normalize_mat2(arma::mat& X, unsigned int p = 1, unsigned int dim = 0) {
    dim = (dim == 2) ? 0 : dim;
    arma::mat X_norm = actionet::normalizeMatrix(X, p, dim);

    return (X_norm);
}

// [[Rcpp::export]]
arma::sp_mat normalize_spmat2(arma::sp_mat& X, unsigned int p = 1, unsigned int dim = 0) {
    dim = (dim == 2) ? 0 : dim;
    arma::sp_mat X_norm = actionet::normalizeMatrix(X, p, dim);

    return (X_norm);
}

// [[Rcpp::export]]
arma::sp_mat normalizeGraph(arma::sp_mat& G, int norm_type = 0) {
    arma::sp_mat G_norm = actionet::normalizeGraph(G, norm_type);
    return (G_norm);
}
