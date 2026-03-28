// Singular value decomposition (SVD) using Feng method
// From: Xu Feng, Yuyang Xie, and Yaohang Li, "Fast Randomzied SVD for Sparse
// Data," in Proc. the 10th Asian Conference on Machine Learning (ACML),
// Beijing, China, Nov. 2018.
#ifndef ACTIONET_SVD_FENG_HPP
#define ACTIONET_SVD_FENG_HPP

#include "libactionet_config.hpp"
#include "decomposition/matrix_operator.hpp"

namespace actionet {

/// @brief Randomized SVD using the Feng method.
///
/// Computes a rank-@p dim approximation via the Feng sparse-aware randomized
/// algorithm.  Particularly efficient for large sparse matrices where the
/// sparsity pattern can be exploited during the sketch phase.
///
/// @tparam T Dense (arma::mat) or sparse (arma::sp_mat) matrix type.
/// @param A      Input matrix (m × n).
/// @param dim    Number of singular vectors/values to compute.
/// @param max_it Maximum number of iterations (default 5).
/// @param seed   Random seed (0 = non-deterministic).
/// @param verbose Print progress messages if true.
///
/// @return Field containing {U (m×dim), sigma (dim), V (n×dim)}.
template <typename T>
arma::field<arma::mat> svdFeng(const T& A, int dim, int max_it = 5, int seed = 0, bool verbose = true);

/// @brief Randomized SVD using the Feng method via a matrix operator backend.
///
/// Identical algorithm to the matrix overload but delegates all matrix-vector
/// products to @p A.matvec / @p A.rmatvec.  Use when the full matrix is not
/// materialised in memory.
///
/// @param A       Matrix operator representing an m × n matrix.
/// @param dim     Number of singular vectors/values to compute.
/// @param max_it  Maximum number of iterations (default 5).
/// @param seed    Random seed (0 = non-deterministic).
/// @param verbose Print progress messages if true.
///
/// @return Field containing {U (m×dim), sigma (dim), V (n×dim)}.
arma::field<arma::mat> svdFeng(const MatrixOperator& A, int dim, int max_it = 5,
                               int seed = 0, bool verbose = true);

} // namespace actionet

#endif //ACTIONET_SVD_FENG_HPP
