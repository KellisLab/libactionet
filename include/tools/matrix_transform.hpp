#ifndef ACTIONET_MATRIX_TRANSFORM_HPP
#define ACTIONET_MATRIX_TRANSFORM_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Normalize a matrix by p-norm along a dimension.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param X Input matrix.
    /// @param p P-norm degree (1 = L1, 2 = L2, <=0 returns unnormalized).
    /// @param dim Dimension to normalize (0 = columns, 1 = rows).
    /// @return Normalized matrix.
    template <typename T>
    T normalizeMatrix(T& X, unsigned int p = 1, unsigned int dim = 0);

    /// @brief Scale a matrix by a per-row/column vector.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param X Input matrix.
    /// @param v Scale vector.
    /// @param dim Dimension to scale (0 = columns, 1 = rows).
    /// @return Scaled matrix.
    template <typename T>
    T scaleMatrix(T& X, arma::vec& v, unsigned int dim = 0);

    /// @brief Normalize a graph adjacency matrix in-place.
    ///
    /// @param G Graph adjacency matrix (modified in-place).
    /// @param norm_method 0 = column (pagerank), 1 = row, 2 = sym_pagerank.
    void normalizeGraph(arma::sp_mat& G, int norm_method = 1);

    /// @brief Normalize a graph adjacency matrix in-place, also returning the
    ///        pre-normalization column sums.
    ///
    /// Fused variant intended for callers (e.g. PageRank pre-processing) that
    /// need to identify sink columns before scaling. Avoids a second full
    /// non-zero pass over the input matrix.
    ///
    /// The reported column sums are computed with the same per-branch
    /// accumulation convention that normalizeGraph uses internally:
    ///   - norm_method 0 (column-normalize): sum of absolute values per column.
    ///   - norm_method 1 (row-normalize): sum of absolute values per column
    ///     (computed in a dedicated pass because the internal row-sum pass
    ///     does not visit columns).
    ///   - norm_method 2 (symmetric): plain sum per column, matching the
    ///     symmetric branch's internal accumulation.
    /// For non-negative edge weights (the supported input for PageRank-style
    /// diffusion) all three conventions coincide.
    ///
    /// @param G                 Graph adjacency matrix (modified in-place).
    /// @param norm_method       0 = column (pagerank), 1 = row, 2 = sym_pagerank.
    /// @param col_sums_out      Output vector of length G.n_cols; filled with the
    ///                          pre-normalization column sums of G.
    void normalizeGraph(arma::sp_mat& G, int norm_method, arma::vec& col_sums_out);


    /// @brief Normalize score matrix for downstream scoring.
    ///
    /// @param scores Input score matrix.
    /// @param method Normalization method: 0=none, 1=zscore, 2=robust_zscore, 3=mean_center.
    /// @param thread_no Number of threads (0 = auto).
    /// @return Normalized scores.
    arma::mat normalize_scores(arma::mat scores, int method = 1, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_MATRIX_TRANSFORM_HPP
