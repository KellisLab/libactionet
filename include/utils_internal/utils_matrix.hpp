// Internal helpers for sparse-dense matrix operations
// Uses native Armadillo operations for thread-safety and large matrix support
#ifndef ACTIONET_UTILS_MATRIX_HPP
#define ACTIONET_UTILS_MATRIX_HPP

#include "libactionet_config.hpp"

// Sparse matrix-dense vector product: y = A*x
arma::vec spmat_vec_product(const arma::sp_mat& A, const arma::vec& x);

// Thread-safe parallel sparse-dense matrix product: C = A*B
arma::mat spmat_mat_product_parallel(const arma::sp_mat& A, const arma::mat& B, int thread_no);


#endif //ACTIONET_UTILS_MATRIX_HPP
