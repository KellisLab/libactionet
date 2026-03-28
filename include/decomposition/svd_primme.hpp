// Singular value decomposition (SVD) using PRIMME_SVDS
#ifndef ACTIONET_SVD_PRIMME_HPP
#define ACTIONET_SVD_PRIMME_HPP

#include "libactionet_config.hpp"

namespace actionet {

/// @brief Truncated SVD using PRIMME_SVDS (sparse matrices).
///
/// PRIMME_SVDS is an implicitly restarted Lanczos solver designed for large
/// sparse matrices.  It supports 64-bit indexing and is preferred over IRLBA
/// for matrices with more than 2^31 non-zero elements.
///
/// @param A       Sparse input matrix (m × n).
/// @param k       Number of singular vectors/values to compute.
/// @param max_it  Maximum number of iterations (0 = auto).
/// @param seed    Random seed for initialisation.
/// @param verbose Print progress messages if true.
///
/// @return Field containing {U (m×k), sigma (k), V (n×k)}.
arma::field<arma::mat> svdPRIMME(const arma::sp_mat& A, int k, int max_it = 1000, int seed = 0, bool verbose = true);

/// @brief Truncated SVD using PRIMME_SVDS (dense matrices).
///
/// Convenience overload that wraps a dense matrix with the same PRIMME_SVDS
/// iterative solver.  For small-to-medium dense matrices the randomised
/// methods (Halko, Feng) are typically faster; PRIMME is preferred only when
/// very high accuracy is required or the matrix is extremely large.
///
/// @param A       Dense input matrix (m × n).
/// @param k       Number of singular vectors/values to compute.
/// @param max_it  Maximum number of iterations (0 = auto).
/// @param seed    Random seed for initialisation.
/// @param verbose Print progress messages if true.
///
/// @return Field containing {U (m×k), sigma (k), V (n×k)}.
arma::field<arma::mat> svdPRIMME(const arma::mat& A, int k, int max_it = 1000, int seed = 0, bool verbose = true);

} // namespace actionet

#endif // ACTIONET_SVD_PRIMME_HPP
