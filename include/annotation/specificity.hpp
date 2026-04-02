#ifndef ACTIONET_SPECIFICITY_HPP
#define ACTIONET_SPECIFICITY_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Compute feature specificity scores using Bernstein-type tail bounds.
    ///
    /// Identifies features whose expression is enriched or depleted in each
    /// group defined by the membership matrix H.  The algorithm:
    ///   1. Shifts S so all values are non-negative (subtracts global min).
    ///      S itself is **not** modified; a shifted copy is used internally.
    ///   2. Normalises H column-wise (divides each column by its mean).
    ///   3. Computes per-gene and per-cell density estimates (row_p, col_p)
    ///      and observed co-occurrence Obs = S.t() * H_norm  (genes × k).
    ///   4. Derives expected co-occurrence and variance under a null model,
    ///      then evaluates one-sided Bernstein tail bounds to yield log10-scaled
    ///      significance matrices.
    ///
    /// @note S is **not** modified.  The function operates on an internal shifted
    ///       copy so the caller's matrix is left unchanged.
    ///
    /// @tparam T  arma::mat or arma::sp_mat.
    /// @param S   Feature matrix (cells × genes, i.e. obs × var).
    /// @param H   Group membership / archetype weight matrix (cells × k).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field(3):
    ///   - (0) average_profile  (genes × k)
    ///   - (1) upper_significance (genes × k) -- enrichment log10 p-values
    ///   - (2) lower_significance (genes × k) -- depletion  log10 p-values
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(const T& S, const arma::mat& H, int thread_no = 0);

    /// @brief Compute feature specificity from discrete cluster labels.
    ///
    /// Converts 1-based integer labels into a binary membership matrix and
    /// delegates to the matrix overload above.
    ///
    /// @tparam T  arma::mat or arma::sp_mat.
    /// @param S      Feature matrix (cells × genes, i.e. obs × var).
    /// @param labels  Cluster labels (1-based, length = n_cells).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Same as the matrix overload.
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(const T& S, const arma::uvec& labels, int thread_no = 0);

    // Forward-declare the backed sparse operator so we can declare the backed
    // overloads without pulling the full HDF5 headers into every translation
    // unit that includes specificity.hpp.
    class BackedSparseMatrixOperator;

    /// @brief Compute feature specificity using an HDF5-backed sparse matrix.
    ///
    /// Performs the same Bernstein-tail scoring as the in-memory overloads but
    /// reads the expression matrix in chunks directly from the h5ad file.  The
    /// algorithm executes in a single streaming pass over the stored non-zero
    /// entries, accumulating all required statistics before applying the min-shift
    /// correction and computing the tail bounds.
    ///
    /// Unlike the in-memory template overloads this function is non-mutating:
    /// the underlying HDF5 data is never modified.
    ///
    /// @param op        Backed sparse matrix operator (obs × var, i.e. cells × genes).
    /// @param H         Group membership / archetype weight matrix (cells × k).
    /// @param thread_no Thread hint for internal OpenMP loops (0 = auto).
    ///
    /// @return Same field layout as the in-memory overloads.
    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::mat& H, int thread_no = 0);

    /// @brief Compute backed feature specificity from discrete cluster labels.
    ///
    /// Converts 1-based integer labels into a binary membership matrix and
    /// delegates to the backed matrix overload above.
    ///
    /// @param op        Backed sparse matrix operator (obs × var, i.e. cells × genes).
    /// @param labels    Cluster labels (1-based, length = n_cells).
    /// @param thread_no Thread hint for internal OpenMP loops (0 = auto).
    ///
    /// @return Same field layout as the in-memory overloads.
    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no = 0);

    // Forward-declare the backed dense operator.
    class BackedDenseMatrixOperator;

    /// @brief Compute feature specificity using an HDF5-backed dense matrix.
    ///
    /// Reads the dense expression matrix in obs-chunks directly from the h5ad
    /// file.  Accumulates global minimum, per-gene density and column sums in
    /// one streaming pass, applies the min-shift analytically, then computes
    /// `Obs = S.t() * H_norm` via the operator's `rmatmat` for the full dataset
    /// before evaluating the Bernstein tail bounds.
    ///
    /// The underlying HDF5 data is never modified.
    ///
    /// @param op        Backed dense matrix operator (obs × var, i.e. cells × genes).
    /// @param H         Group membership / archetype weight matrix (cells × k).
    /// @param thread_no Thread hint for internal OpenMP loops (0 = auto).
    ///
    /// @return Same field layout as the in-memory overloads.
    arma::field<arma::mat> computeFeatureSpecificity(BackedDenseMatrixOperator& op,
                                                     const arma::mat& H, int thread_no = 0);

    /// @brief Compute backed-dense feature specificity from discrete cluster labels.
    ///
    /// Converts 1-based integer labels into a binary membership matrix and
    /// delegates to the backed-dense matrix overload above.
    ///
    /// @param op        Backed dense matrix operator (obs × var, i.e. cells × genes).
    /// @param labels    Cluster labels (1-based, length = n_cells).
    /// @param thread_no Thread hint for internal OpenMP loops (0 = auto).
    ///
    /// @return Same field layout as the in-memory overloads.
    arma::field<arma::mat> computeFeatureSpecificity(BackedDenseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_SPECIFICITY_HPP
