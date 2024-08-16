// Postprocess ACTION output
#ifndef ACTIONET_ACTION_POST_HPP
#define ACTIONET_ACTION_POST_HPP

#include "libactionet_config.hpp"

// Exported
namespace ACTIONet {

    // To store the output of prune_archetypes()
    struct multilevel_archetypal_decomposition {
        arma::uvec selected_archs; // If hub removal requested, this will hold the indices
        // of retained archetypes
        arma::mat C_stacked;       // Stacking of C matrices, after potentially removing the hub
        // archetypes
        arma::mat H_stacked;       // Stacking of H matrices, after potentially removing the hub
        // archetypes
    };

    // To store the output of unify_archetypes()
    struct unification_results {
        arma::mat dag_adj;
        arma::vec dag_node_annotations;
        arma::uvec selected_archetypes;
        arma::mat C_unified;
        arma::mat H_unified;
        arma::uvec assigned_archetypes;
        arma::vec archetype_group;
        arma::mat arch_membership_weights;
    };

    // Functions
    multilevel_archetypal_decomposition
    prune_archetypes(arma::field<arma::mat> C_trace, arma::field<arma::mat> H_trace, double min_specificity_z_threshold,
                     int min_cells = 3);

    unification_results
    unify_archetypes(arma::mat &S_r, arma::mat &C_stacked, arma::mat &H_stacked, double backbone_density,
                     double resolution, int min_cluster_size, int thread_no, int normalization);

} // namespace ACTIONet

#endif //ACTIONET_ACTION_POST_HPP
