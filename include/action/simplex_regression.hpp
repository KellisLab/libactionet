// Simplex regression algorithm
#ifndef LIBACTIONET_SIMPLEX_REGRESSION_HPP
#define LIBACTIONET_SIMPLEX_REGRESSION_HPP

#include "libactionet_config.hpp"
#include <cblas.h>
#include <cassert>

//Functions: private

//Active-Set Method with direct inversion, with update (matrix inversion lemma)
arma::vec activeSet_arma(arma::mat &M, arma::vec &b, double lambda2, double epsilon);

// Active-Set Method with direct inversion, with update (matrix inversion lemma)
// Memorize M.double* M + lam2sq = G
arma::vec activeSetS_arma(arma::mat &M, arma::vec &b, arma::mat &G, double lambda2, double epsilon);

// void activeSet_arma_ptr(double* M_ptr, int m, int n, double* b_ptr, double* x_ptr);

// Exported
namespace ACTIONet {

    // Simplex regression for AA: min_{X} (|| AX - B ||) s.t. simplex constraint using ACTIVE Set Method
    arma::mat run_simplex_regression(arma::mat &A, arma::mat &B, bool computeXtX = false);

} // namespace ACTIONet

#endif //LIBACTIONET_SIMPLEX_REGRESSION_HPP
