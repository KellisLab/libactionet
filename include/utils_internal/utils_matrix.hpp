// Internal helpers for core matrix operations
#ifndef ACTIONET_UTILS_MATRIX_HPP
#define ACTIONET_UTILS_MATRIX_HPP

#include "libactionet_config.hpp"
#include <cholmod.h>

void
dsdmult(char transpose, int n_rows, int n_cols, const void *A, const double *x, double *out, cholmod_common *chol_cp);

cholmod_sparse *as_cholmod_sparse(const arma::sp_mat &A, cholmod_sparse *chol_A, cholmod_common *chol_c);

arma::sp_mat &as_arma_sparse(const cholmod_sparse *chol_A, arma::sp_mat &A, cholmod_common *chol_c);

arma::vec spmat_vec_product(const arma::sp_mat &A, arma::vec &x);

arma::mat spmat_mat_product(const arma::sp_mat &A, arma::mat &B);

// TODO: REMOVE?
arma::sp_mat spmat_spmat_product(const arma::sp_mat &A, const arma::sp_mat &B);

arma::mat spmat_mat_product_parallel(const arma::sp_mat &A, arma::mat &B, int thread_no);

#endif //ACTIONET_UTILS_MATRIX_HPP
