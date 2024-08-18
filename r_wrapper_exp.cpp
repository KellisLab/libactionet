// Rcpp interface for libactionet
// [[Rcpp::interfaces(r, cpp)]]

// Enable build configuration for R interface
#define LIBACTIONET_BUILD_R
// Header `libactionet.hpp` configures package and includes `RcppArmadillo.h`. It must precede `Rcpp.h`.
#include "libactionet.hpp"
#include "Rcpp.h"
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
Rcpp::List IRLB_SVD2(arma::sp_mat &A, int dim, int iters = 1000, int seed = 0, int verbose = 1) {
    arma::field<arma::mat> SVD_out = actionet::IRLB_SVD(A, dim, iters, seed, verbose);

    Rcpp::List res;

    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}

// [[Rcpp::export]]
Rcpp::List IRLB_SVD2(arma::mat &A, int dim, int iters = 1000, int seed = 0, int verbose = 1) {
    arma::field<arma::mat> SVD_out = actionet::IRLB_SVD(A, dim, iters, seed, verbose);

    Rcpp::List res;

    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}