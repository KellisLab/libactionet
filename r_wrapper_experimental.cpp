// Rcpp interface for libactionet
// [[Rcpp::interfaces(r, cpp)]]

// Enable build configuration for R interface
#define LIBACTIONET_BUILD_R
// Header `libactionet.hpp` configures package and includes `RcppArmadillo.h`. It must precede `Rcpp.h`.
#include "libactionet.hpp"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List runSVD(arma::sp_mat &S, int k = 30, int iter = 0, int seed = 0, int algorithm = 0, int verbose = 1) {
    if (iter == 0) {
        switch (algorithm) {
            case 1: iter = 1000;
                break;
            default: iter = 5;
                break;
        }
    }

    arma::field<arma::mat> SVD_out = actionet::runSVD(S, k, iter, seed, algorithm, verbose);

    Rcpp::List res;

    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}

// [[Rcpp::export]]
Rcpp::List runSVD_full(arma::mat &S, int k = 30, int iter = 0, int seed = 0, int algorithm = 0, int verbose = 1) {
    if (iter == 0) {
        switch (algorithm) {
            case 1: iter = 1000;
                break;
            default: iter = 5;
                break;
        }
    }

    arma::field<arma::mat> SVD_out = actionet::runSVD(S, k, iter, seed, algorithm, verbose);

    Rcpp::List res;

    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}
