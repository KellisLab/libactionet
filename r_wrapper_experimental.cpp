// Rcpp interface for libactionet
// [[Rcpp::interfaces(r, cpp)]]

// Enable build configuration for R interface
#define LIBACTIONET_BUILD_R
// Header `libactionet.hpp` configures package and includes `RcppArmadillo.h`. It must precede `Rcpp.h`.
#include "libactionet.hpp"
// [[Rcpp::depends(RcppArmadillo)]]

//' Computes SVD decomposition
//'
//' This is direct implementation of the randomized SVD algorithm:
//' From: IRLBA R Package
//'
//' @param A Input matrix ("sparseMatrix")
//' @param k Dimension of SVD decomposition
//' @param max_it Number of iterations (default=5)
//' @param seed Random seed (default=0)
//' @param algorithm SVD algorithm to use. Currently supported methods are blah blah blah
//'
//' @return A named list with U, sigma, and V components
//'
//' @examples
//' A = randn(100, 20)
//' svd.out = runSVD(A, dim = 3)
//' U = svd.out$u
// [[Rcpp::export]]
Rcpp::List runSVD(arma::sp_mat &A, int k = 30, int max_it = 0, int seed = 0, int algorithm = 0, int verbose = 1) {
    arma::field<arma::mat> SVD_out = actionet::runSVD(A, k, max_it, seed, algorithm, verbose);

    Rcpp::List res;
    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}

// [[Rcpp::export]]
Rcpp::List runSVD_full(arma::mat &A, int k = 30, int max_it = 0, int seed = 0, int algorithm = 0, int verbose = 1) {
    arma::field<arma::mat> SVD_out = actionet::runSVD(A, k, max_it, seed, algorithm, verbose);

    Rcpp::List res;
    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}
