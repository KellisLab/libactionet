// Internal helpers for sparse-dense matrix operations
// Uses native Armadillo operations for thread-safety and large matrix support
#ifndef ACTIONET_UTILS_MATRIX_HPP
#define ACTIONET_UTILS_MATRIX_HPP

#include "libactionet_config.hpp"

// Sparse matrix-dense vector product: y = A*x
arma::vec spmat_vec_product(const arma::sp_mat& A, const arma::vec& x);

// Sparse matrix-dense vector product with transpose option: y = A*x or y = A'*x
arma::vec spmat_vec_product_t(const arma::sp_mat& A, const arma::vec& x, bool transpose);

// Sparse matrix-dense matrix product: C = A*B
arma::mat spmat_mat_product(const arma::sp_mat& A, const arma::mat& B);

// Thread-safe parallel sparse-dense matrix product
arma::mat spmat_mat_product_parallel(const arma::sp_mat& A, const arma::mat& B, int thread_no);

// Sparse-sparse matrix product: C = A*B
arma::sp_mat spmat_spmat_product(const arma::sp_mat& A, const arma::sp_mat& B);


#endif //ACTIONET_UTILS_MATRIX_HPP
