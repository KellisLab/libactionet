#ifndef ACTIONET_SPECIFICITY_HPP
#define ACTIONET_SPECIFICITY_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Compute feature specificity from archetype weights.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param S Feature matrix (features x cells).
    /// @param H Archetype weights (k x cells).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field with average profiles and tail significance matrices.
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(T& S, arma::mat& H, int thread_no = 0);

    /// @brief Compute feature specificity from discrete labels.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param S Feature matrix (features x cells).
    /// @param labels Cluster labels (1-based).
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Field with average profiles and tail significance matrices.
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(T& S, arma::uvec& labels, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_SPECIFICITY_HPP
