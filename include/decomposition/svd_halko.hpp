// Singular value decomposition (SVD) with Halko method
// Implemented from: N Halko, P. G Martinsson, and J. A Tropp. Finding structure with
// randomness: Probabilistic algorithms for constructing approximate matrix
// decompositions. Siam Review, 53(2):217-288, 2011.
#ifndef ACTIONET_SVD_HALKO_HPP
#define ACTIONET_SVD_HALKO_HPP

#include "libactionet_config.hpp"
#include "decomposition/matrix_operator.hpp"

namespace actionet {

/// @brief Randomized SVD using the Halko method.
///
/// Computes a rank-@p dim approximation of @p A via randomized range finding
/// followed by exact SVD of the low-dimensional sketch.  Power iterations
/// improve accuracy at the cost of additional matrix-vector products.
///
/// @tparam T Dense (arma::mat) or sparse (arma::sp_mat) matrix type.
/// @param A       Input matrix (m × n).
/// @param dim     Number of singular vectors/values to compute.
/// @param iters   Number of power iterations (default 5; higher = more accurate, slower).
/// @param seed    Random seed for the sketch (0 = non-deterministic).
/// @param verbose Print progress messages if true.
///
/// @return Field containing {U (m×dim), sigma (dim), V (n×dim)}.
template <typename T>
arma::field<arma::mat> svdHalko(const T& A, int dim, int iters = 5, int seed = 0, bool verbose = true);

/// @brief Randomized SVD using the Halko method via a matrix operator backend.
///
/// Identical algorithm to the matrix overload but delegates all matrix-vector
/// products to @p A.matvec / @p A.rmatvec.  Use when the full matrix is not
/// materialised in memory (out-of-core or Python-backed arrays).
///
/// @param A       Matrix operator representing an m × n matrix.
/// @param dim     Number of singular vectors/values to compute.
/// @param iters   Number of power iterations (default 5).
/// @param seed    Random seed (0 = non-deterministic).
/// @param verbose Print progress messages if true.
///
/// @return Field containing {U (m×dim), sigma (dim), V (n×dim)}.
arma::field<arma::mat> svdHalko(const MatrixOperator& A, int dim, int iters = 5,
                                int seed = 0, bool verbose = true);

} // namespace actionet

#endif //ACTIONET_SVD_HALKO_HPP
