// Postprocess ACTION output
#ifndef ACTIONET_ACTION_POST_HPP
#define ACTIONET_ACTION_POST_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Structs
    /// @brief Stores the output of <code>collectArchetypes()</code>.
    ///
    /// Members:
    /// - <b>selected_archs</b>: Indices of retained archetypes.
    /// - <b>C_stacked</b>: Row-wise concatenation of filtered C matrices.
    /// - <b>H_stacked</b>: Column-wise concatenation of filtered H matrices.
    struct ResCollectArch {
        arma::uvec selected_archs; // If hub removal requested, this will hold the indices
        // of retained archetypes
        arma::mat C_stacked; // Stacking of C matrices, after potentially removing the hub
        // archetypes
        arma::mat H_stacked; // Stacking of H matrices, after potentially removing the hub
        // archetypes
    };

    /// @brief Stores the output of <code>mergeArchetypes()</code>.
    ///
    /// Members:
    /// - <b>selected_archetypes</b>: Indices of representative archetypes.
    /// - <b>C_merged</b>: Reduced representative C matrix.
    /// - <b>H_merged</b>: Reduced representative H matrix.
    /// - <b>assigned_archetypes</b>: Assignment per observation.
    struct ResMergeArch {
        arma::uvec selected_archetypes;
        arma::mat C_merged;
        arma::mat H_merged;
        arma::uvec assigned_archetypes;
    };

    /// @brief Filter and aggregate multi-level archetypes.
    ///
    /// @param C_trace Field of C matrices from <code>runACTION()</code>.
    /// @param H_trace Field of H matrices from <code>runACTION()</code>.
    /// @param spec_th Specificity threshold (z-score).
    /// @param min_obs Minimum observations per archetype.
    ///
    /// @return <code>ResCollectArch</code> with stacked C/H and selected indices.
    ResCollectArch
        collectArchetypes(arma::field<arma::mat>& C_trace, arma::field<arma::mat>& H_trace, double spec_th = -3,
                          int min_obs = 3);

    /// @brief Identify and merge redundant archetypes into a representative subset.
    ///
    /// @param S_r Reduced data matrix (<em>vars</em> x <em>obs</em>).
    /// @param C_stacked Filtered C matrix from <code>collectArchetypes()</code>.
    /// @param H_stacked Filtered H matrix from <code>collectArchetypes()</code>.
    /// @param thread_no Number of CPU threads (0 = auto).
    ///
    /// @return <code>ResMergeArch</code> with merged results and assignments.
    ResMergeArch
        mergeArchetypes(arma::mat& S_r, arma::mat& C_stacked, arma::mat& H_stacked, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_POST_HPP
