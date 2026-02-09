#ifndef ACTIONET_UTILS_STATS_HPP
#define ACTIONET_UTILS_STATS_HPP

#include "libactionet_config.hpp"

/// @brief Z-score normalize along a dimension.
///
/// @param A Input matrix.
/// @param dim Dimension (0 = columns, 1 = rows).
/// @param thread_no Number of threads (0 = auto).
/// @return Z-scored matrix.
arma::mat zscore(arma::mat &A, int dim = 0, int thread_no = 1);

/// @brief Robust z-score normalization.
///
/// @param A Input matrix.
/// @param dim Dimension (0 = columns, 1 = rows).
/// @param thread_no Number of threads (0 = auto).
/// @return Robust z-scored matrix.
arma::mat robust_zscore(arma::mat &A, int dim = 0, int thread_no = 1);

/// @brief Tied z-score helper (legacy).
arma::mat tzscoret(arma::mat &A);

/// @brief Mean-center a matrix by columns.
arma::mat mean_center(const arma::mat &A);

#endif //ACTIONET_UTILS_STATS_HPP
