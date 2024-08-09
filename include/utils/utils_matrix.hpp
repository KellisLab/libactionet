#ifndef UTILS_MATRIX_HPP
#define UTILS_MATRIX_HPP

#include "config_arma.hpp"
#include "config_interface.hpp"

#include "utils_parallel.hpp"
#include "utils_math.hpp"

#include <cholmod.h>

void dsdmult(char transpose, int n_rows, int n_cols, const void *A, const double *x, double *out, cholmod_common *chol_cp);

cholmod_sparse *as_cholmod_sparse(const arma::sp_mat &A, cholmod_sparse *chol_A, cholmod_common *chol_c);

arma::sp_mat &as_arma_sparse(cholmod_sparse *chol_A, arma::sp_mat &A, cholmod_common *chol_c);

arma::mat normalize_mat(arma::mat &X, int normalization, int dim);

arma::sp_mat normalize_mat(arma::sp_mat &X, int normalization, int dim);

arma::vec spmat_vec_product(arma::sp_mat &A, arma::vec &x);

arma::mat spmat_mat_product(arma::sp_mat &A, arma::mat &B);

// arma::sp_mat spmat_spmat_product(arma::sp_mat &A, arma::sp_mat &B);

arma::mat spmat_mat_product_parallel(arma::sp_mat &A, arma::mat &B, int thread_no);

arma::mat mat_mat_product_parallel(arma::mat &A, arma::mat &B, int thread_no);

#endif
