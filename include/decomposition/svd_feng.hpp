// Singular value decomposition (SVD) using Feng method
// From: Xu Feng, Yuyang Xie, and Yaohang Li, "Fast Randomzied SVD for Sparse
// Data," in Proc. the 10th Asian Conference on Machine Learning (ACML),
// Beijing, China, Nov. 2018.
#ifndef ACTIONET_SVD_FENG_HPP
#define ACTIONET_SVD_FENG_HPP

#include "libactionet_config.hpp"
#include "decomposition/matrix_operator.hpp"

/// @brief Randomized SVD using the Feng method.
///
/// @tparam T Dense or sparse matrix type.
/// @param A Input matrix.
/// @param dim Number of components.
/// @param max_it Maximum number of iterations.
/// @param seed Random seed.
/// @param verbose Print progress messages.
///
/// @return Field containing {U, S, V}.
template <typename T>
arma::field<arma::mat> svdFeng(T& A, int dim, int max_it = 5, int seed = 0, bool verbose = true);

/// @brief Randomized SVD using Feng with a matrix operator backend.
arma::field<arma::mat> svdFeng(const actionet::MatrixOperator& A, int dim, int max_it = 5,
                               int seed = 0, bool verbose = true);

#endif //ACTIONET_SVD_FENG_HPP
