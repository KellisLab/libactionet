#ifndef SIMPLEX_REGRESSION_HPP
#define SIMPLEX_REGRESSION_HPP

#include "action.hpp"

#include <cblas.h>
#include <cassert>

/* **************************
 * Active-Set Method with direct inversion, with update(matrix inversion lemma)
 * **************************/
arma::vec activeSet_arma(arma::mat &M, arma::vec &b, double lambda2, double epsilon);

/// Active-Set Method with direct inversion, with update(matrix inversion lemma)
/// Memorize M.double* M + lam2sq = G
arma::vec activeSetS_arma(arma::mat &M, arma::vec &b, arma::mat &G, double lambda2, double epsilon);

// void activeSet_arma_ptr(double* M_ptr, int m, int n, double* b_ptr, double* x_ptr);

// namespace ACTIONet {

// min(|| AX - B ||) s.t. simplex constraint
// mat run_simplex_regression(mat& A, mat& B, bool computeXtX);

// }  // namespace ACTIONet

#endif
