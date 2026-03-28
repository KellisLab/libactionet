// Singular value decomposition (SVD) using IRLBA
// Implemented from irlba R package (https://github.com/bwlewis/irlba)
#ifndef ACTIONET_SVD_IRBLA_HPP
#define ACTIONET_SVD_IRBLA_HPP

#include "libactionet_config.hpp"
#include "decomposition/matrix_operator.hpp"

namespace actionet {

/// @brief Truncated SVD using IRLBA (sparse).
///
/// @param A Sparse input matrix.
/// @param dim Number of components.
/// @param iters Maximum iterations.
/// @param seed Random seed.
/// @param verbose Print progress messages.
///
/// @return Field containing {U, S, V}.
arma::field<arma::mat> svdIRLB(const arma::sp_mat& A, int dim, int iters = 1000, int seed = 0, bool verbose = true);

/// @brief Truncated SVD using IRLBA (dense).
///
/// @param A Dense input matrix.
/// @param dim Number of components.
/// @param iters Maximum iterations.
/// @param seed Random seed.
/// @param verbose Print progress messages.
///
/// @return Field containing {U, S, V}.
arma::field<arma::mat> svdIRLB(const arma::mat& A, int dim, int iters = 1000, int seed = 0, bool verbose = true);

/// @brief Truncated SVD using IRLBA via abstract matrix operator.
///
/// Enables IRLB on out-of-memory / backed matrices without materialising the
/// full matrix.  Matrix-vector products are delegated to the operator's
/// matvec (y = A*x) and rmatvec (y = A'*x) methods.
///
/// @param A Matrix operator (m × n logical matrix).
/// @param dim Number of components.
/// @param iters Maximum iterations.
/// @param seed Random seed.
/// @param verbose Print progress messages.
///
/// @return Field containing {U, S, V}.
arma::field<arma::mat> svdIRLB(const MatrixOperator& A, int dim,
                                int iters = 1000, int seed = 0, bool verbose = true);

} // namespace actionet

#endif //ACTIONET_SVD_IRBLA_HPP
