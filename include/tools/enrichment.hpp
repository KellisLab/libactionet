#ifndef ACTIONET_ENRICHMENT_HPP
#define ACTIONET_ENRICHMENT_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Compute label enrichment scores on a graph.
    ///
    /// @param G Graph adjacency matrix.
    /// @param scores Scores matrix (cells x labels).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Log p-value matrix (cells x labels).
    arma::mat computeGraphLabelEnrichment(const arma::sp_mat& G, const arma::mat& scores, int thread_no = 0);

    /// @brief Assess enrichment of scores against binary associations.
    ///
    /// For each (score column j, association column k) pair, computes the
    /// maximum Bennett-inequality log-p-value achieved by sliding a cutoff
    /// down the descending-rank order of scores.col(j) and restricting
    /// the observed mass to features flagged in associations.col(k).
    ///
    /// The `associations` matrix is treated as boolean (its nonzero pattern);
    /// weights are ignored. A local copy is taken so the caller's argument
    /// is not modified.
    ///
    /// @param scores        Feature scores (features x conditions).
    /// @param associations  Binary associations (features x gene sets). Not modified.
    /// @param thread_no     Number of threads (0 = auto).
    ///
    /// @return Field of two matrices:
    ///         - `field(0) = logPvals` (n_gene_sets x n_conditions).
    ///         - `field(1) = peak_rank_idx` (n_gene_sets x n_conditions).
    ///           Each entry is the 0-based position, in the descending
    ///           sort of the corresponding score column, at which the
    ///           logPval peaks. NOT a score threshold; to recover the
    ///           score at that position, index into
    ///           `arma::sort(scores.col(j), "descend")` with the value.
    arma::field<arma::mat> assess_enrichment(const arma::mat& scores, const arma::sp_mat& associations, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ENRICHMENT_HPP
