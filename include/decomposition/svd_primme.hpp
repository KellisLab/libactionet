// Singular value decomposition (SVD) using PRIMME_SVDS
#ifndef ACTIONET_SVD_PRIMME_HPP
#define ACTIONET_SVD_PRIMME_HPP

#include "libactionet_config.hpp"

/**
 * @brief Compute SVD using PRIMME_SVDS (sparse matrices)
 *
 * PRIMME_SVDS is designed for large sparse matrices and supports 64-bit indexing,
 * making it suitable for matrices with >2^31 non-zero elements.
 *
 * @param A Sparse matrix (genes × cells or similar)
 * @param k Number of singular values/vectors to compute
 * @param max_it Maximum number of iterations (0 = auto)
 * @param seed Random seed for initialization
 * @param verbose Print progress messages
 * @return field<mat> [U, S, V] where A ≈ U * diag(S) * V'
 */
arma::field<arma::mat> svdPRIMME(arma::sp_mat& A, int k, int max_it = 1000, int seed = 0, bool verbose = true);

/**
 * @brief Compute SVD using PRIMME_SVDS (dense matrices)
 *
 * @param A Dense matrix
 * @param k Number of singular values/vectors to compute
 * @param max_it Maximum number of iterations (0 = auto)
 * @param seed Random seed for initialization
 * @param verbose Print progress messages
 * @return field<mat> [U, S, V] where A ≈ U * diag(S) * V'
 */
arma::field<arma::mat> svdPRIMME(arma::mat& A, int k, int max_it = 1000, int seed = 0, bool verbose = true);

#endif // ACTIONET_SVD_PRIMME_HPP
