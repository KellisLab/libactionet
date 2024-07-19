#ifndef HELPERS_MATRIX_HPP
#define HELPERS_MATRIX_HPP

#include "config_arma.hpp"
#include "config_actionet.hpp"

#include <cholmod.h>

void dsdmult(char transpose, int n_rows, int n_cols, const void *A, const double *x, double *out, cholmod_common *chol_cp);

cholmod_sparse *as_cholmod_sparse(const arma::sp_mat &A, cholmod_sparse *chol_A, cholmod_common *chol_c);

arma::sp_mat &as_arma_sparse(cholmod_sparse *chol_A, arma::sp_mat &A, cholmod_common *chol_c);

#endif
