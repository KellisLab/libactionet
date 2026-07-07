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
    /// @param scores Feature scores (features x conditions).
    /// @param associations Binary associations (features x gene sets).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field containing logPvals and thresholds.
    arma::field<arma::mat> assess_enrichment(const arma::mat& scores, arma::sp_mat& associations, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ENRICHMENT_HPP
