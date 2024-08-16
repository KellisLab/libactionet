#ifndef ACTIONET_SIMPLEX_REGRESSION_PRIV_HPP
#define ACTIONET_SIMPLEX_REGRESSION_PRIV_HPP

#include "libactionet_config.hpp"
//#include <cblas.h>
//#include <cassert>

//Functions: private

//Active-Set Method with direct inversion, with update (matrix inversion lemma)
arma::vec activeSet_arma(arma::mat &M, arma::vec &b, double lambda2 = double(1e-5), double epsilon = double(1e-5));

// Active-Set Method with direct inversion, with update (matrix inversion lemma)
// Memorize M.double* M + lam2sq = G
arma::vec activeSetS_arma(arma::mat &M, arma::vec &b, arma::mat &G, double lambda2 = 1e-5, double epsilon = 1e-5);

// void activeSet_arma_ptr(double* M_ptr, int m, int n, double* b_ptr, double* x_ptr);

#endif //ACTIONET_SIMPLEX_REGRESSION_PRIV_HPP
