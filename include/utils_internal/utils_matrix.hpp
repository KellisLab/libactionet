// Internal helpers for sparse-dense matrix operations
// Uses native Armadillo operations for thread-safety and large matrix support
#ifndef ACTIONET_UTILS_MATRIX_HPP
#define ACTIONET_UTILS_MATRIX_HPP

#include "libactionet_config.hpp"

namespace actionet {

// Thread-safe parallel sparse-dense matrix product: C = A*B
/// @brief Multiply sparse matrix by dense matrix in parallel.
///
/// @param A Sparse matrix.
/// @param B Dense matrix.
/// @param thread_no Number of threads (0 = auto).
/// @return Product matrix.
arma::mat spmat_mat_product_parallel(const arma::sp_mat& A, const arma::mat& B, int thread_no);

} // namespace actionet

#endif //ACTIONET_UTILS_MATRIX_HPP
