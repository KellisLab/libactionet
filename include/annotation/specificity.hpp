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
    ///   2. Normalises H column-wise (divides each column by its mean).
    ///   3. Computes per-feature and per-cell density estimates (row_p, col_p)
    ///      and observed co-occurrence Obs = S * H_norm.
    ///   4. Derives expected co-occurrence and variance under a null model,
    ///      then evaluates one-sided Bernstein tail bounds to yield log10-scaled
    ///      significance matrices.
    ///
    /// @note S is modified in-place (values are shifted).  Pass a copy if the
    ///       caller needs the original matrix unchanged.
    ///
    /// @tparam T  arma::mat or arma::sp_mat.
    /// @param S   Feature matrix (features x cells).
    /// @param H   Group membership / archetype weight matrix (k x cells).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field(3):
    ///   - (0) average_profile  (features x k)
    ///   - (1) upper_significance (features x k) -- enrichment log10 p-values
    ///   - (2) lower_significance (features x k) -- depletion  log10 p-values
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(T& S, arma::mat& H, int thread_no = 0);

    /// @brief Compute feature specificity from discrete cluster labels.
    ///
    /// Converts 1-based integer labels into a binary membership matrix and
    /// delegates to the matrix overload above.
    ///
    /// @tparam T  arma::mat or arma::sp_mat.
    /// @param S      Feature matrix (features x cells).
    /// @param labels  Cluster labels (1-based, length = n_cells).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Same as the matrix overload.
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(T& S, arma::uvec& labels, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_SPECIFICITY_HPP
