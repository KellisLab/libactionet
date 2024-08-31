// Postprocess ACTION output
#ifndef ACTIONET_ACTION_POST_HPP
#define ACTIONET_ACTION_POST_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Structs
    /// @brief Stores the output of <code>collect_archetypes()</code>
    ///
    /// Contains the following members:
    /// - <b>selected_archs</b>: Vector containing indices of retained multi-level archetypes.
    /// in [<code>k_min</code>, <code>k_max</code>].
    /// - <b>C_stacked</b>: Matrix constructed by row-wise concatenation of (filtered) <code>C</code> matrices.
    /// - <b>H_stacked</b>: Matrix constructed by column-wise concatenation (filtered) <code>H</code> matrices.
    ///
    /// See <code>run_ACTION()</code>, <code>ResACTION</code>, and <code>run_AA()</code>.
    struct ResCollectArch {
        arma::uvec selected_archs; // If hub removal requested, this will hold the indices
        // of retained archetypes
        arma::mat C_stacked; // Stacking of C matrices, after potentially removing the hub
        // archetypes
        arma::mat H_stacked; // Stacking of H matrices, after potentially removing the hub
        // archetypes
    };

    // To store the output of merge_archetypes()
    /// @brief Stores the output of <code>merge_archetypes()</code>
    ///
    /// Contains the following members:
    /// - <b>selected_archetypes</b>: Vector containing selected representative archetypes determined by SPA.
    /// - <b>C_merged</b>: Reduced representative <b>C</b> matrix.
    /// - <b>H_merged</b>: Reduced representative <b>H</b> matrix.
    /// - <b>assigned_archetypes</b>: Vector of assignments of each data point (rows of <b>C</b>/<b>H</b>) to the
    /// closest representative archetype (<b>selected_archetypes</b>).
    ///
    /// See <code>ResSPA</code>.
    struct ResMergeArch {
        arma::uvec selected_archetypes;
        arma::mat C_merged;
        arma::mat H_merged;
        arma::uvec assigned_archetypes;
    };

    // Functions
    /// @brief Filter and aggregate multi-level archetypes
    ///
    /// @param C_trace Field containing C matrices. Output of <code>run_ACTION()</code> in <code>ResACTION["C"]</code>.
    /// @param H_trace Field containing H matrices. Output of <code>run_ACTION()</code> in <code>ResACTION["H"]</code>.
    /// @param spec_th Minimum threshold (as z-score) to filter archetypes by specificity.
    /// @param min_obs Minimum number of observations assigned to an archetypes needed to retain that archetype.
    ///
    /// @return \c struct of type <code>ResCollectArch</code>.
    ///
    /// See <code>ResCollectArch</code>.
    ResCollectArch
        collect_archetypes(arma::field<arma::mat> C_trace, arma::field<arma::mat> H_trace,
                           double spec_th,
                           int min_obs = 3);

    /// @brief Identify and merge redundant archetypes into a representative subset
    ///
    /// @param S_r Reduced data matrix from which archetypes were found.
    /// @param C_stacked Concatenated (and filtered) <code>C</code> (<b>S_r.n</b> x <em>n</em>) matrix.
    /// Output of <code>collect_archetypes()</code> in <code>ResCollectArch["C_stacked"]</code>.
    /// @param H_stacked Concatenated (and filtered) <code>H</code> (<b>S_r.n</b> x <em>n</em>) matrix.
    /// Output of <code>collect_archetypes()</code> in <code>ResCollectArch["H_stacked"]</code>.
    /// @param normalization Normalization method to apply to <b>S_r</b>.
    /// @param thread_no Number of CPU threads to use. If 0, number is automatically determined.
    ///
    /// @return <code>struct</code> of type <code>ResMergeArch</code>.
    ///
    /// See <code>ResMergeArch</code>.
    ResMergeArch
        merge_archetypes(arma::mat& S_r, arma::mat& C_stacked, arma::mat& H_stacked, int normalization = 0,
                         int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_ACTION_POST_HPP
