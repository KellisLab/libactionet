#include "action/action_post.hpp"
#include "action/spa.hpp"
#include "action/simplex_regression.hpp"
#include "tools/normalization.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"
#include "utils_internal/utils_stats.hpp"

namespace actionet {

    multilevel_archetypal_decomposition prune_archetypes(arma::field<arma::mat> C_trace, arma::field<arma::mat> H_trace,
                                                         double min_specificity_z_threshold, int min_cells) {

        arma::mat C_stacked;
        arma::mat H_stacked;
        int depth = H_trace.size();

        multilevel_archetypal_decomposition results;

        // Vector contains an element for k==0, this have to -1
        stdout_printf("Joining trace of C & H matrices (depth = %d) ... ", depth - 1);
        // Group H and C matrices for different values of k (#archs) into joint matrix
        for (int k = 0; k < depth; k++) {
            if (H_trace[k].n_rows == 0)
                continue;

            if (H_stacked.n_elem == 0) {
                C_stacked = C_trace[k];
                H_stacked = H_trace[k];
            } else {
                C_stacked = arma::join_rows(C_stacked, C_trace[k]);
                H_stacked = arma::join_cols(H_stacked, H_trace[k]);
            }
        }
        int total_archs = H_stacked.n_rows;

        stdout_printf("done (%d archetypes)\n", C_stacked.n_cols);
        FLUSH;
        stdout_printf("Pruning archetypes:\n");
        FLUSH;

        arma::mat backbone = arma::cor(arma::trans(H_stacked));
        backbone.diag().zeros();
        backbone.transform([](double val) { return (val < 0 ? 0 : val); });

        arma::vec pruned = arma::zeros(total_archs);

        // Barrat weighted transitivity: formulation from "Clustering Coefficients for
        // Weighted Networks" (Kalna)
        arma::vec transitivity = arma::zeros(total_archs);
        arma::vec s = arma::sum(backbone, 1); // strength of nodes
        arma::vec d = arma::vec(arma::sum(arma::spones(arma::sp_mat(backbone)), 1));
        for (int k = 0; k < total_archs; k++) {
            double sum = 0;
            for (int i = 0; i < total_archs; i++) {
                double w_ki = backbone(k, i);
                for (int j = 0; j < total_archs; j++) {
                    double w_kj = backbone(k, j);

                    double mean_weight = (w_ki + w_kj) / 2.0;
                    double triangle_mask =
                            backbone(i, k) * backbone(k, j) * backbone(j, i) > 0 ? 1 : 0;

                    sum += mean_weight * triangle_mask;
                }
            }
            transitivity(k) = sum / (s(k) * (d(k) - 1));
        }

        arma::vec transitivity_z = zscore(transitivity);
        arma::uvec nonspecific_idx = arma::find(transitivity_z < min_specificity_z_threshold);
        pruned(nonspecific_idx).ones();
        stdout_printf("\tNon-specific archetypes: %d\n", nonspecific_idx.n_elem);
        FLUSH;

        // Find landmark cells, i.e., closest cells to each multi-level archetype (its
        // projection on to the cell space, ish) stdout_printf("Removing unreliable
        // archetypes (based on the landmark cells) ... ");
        double epsilon = 1e-3;
        int bad_archs = 0;
        arma::vec landmark_cells = -arma::ones(total_archs);
        for (int i = 0; i < total_archs; i++) {
            arma::vec h = arma::trans(H_stacked.row(i));
            arma::vec c = C_stacked.col(i);

            arma::uvec h_landmarks = arma::find((arma::max(h) - h) < epsilon);
            arma::uvec c_landmarks = arma::find(0 < c);
            arma::uvec common_landmarks = arma::intersect(h_landmarks, c_landmarks);

            if (0 < common_landmarks.n_elem) { // They don't agree on any samples!
                landmark_cells(i) = common_landmarks(arma::index_max(c(common_landmarks)));
            } else { // Potentially noisy archetype
                pruned(i) = 1;
                bad_archs++;
            }
        }

        stdout_printf("\tUnreliable archetypes: %d\n", bad_archs);
        FLUSH;

        arma::uvec idx = arma::find(C_stacked > 1e-6);
        arma::mat C_bin = C_stacked;
        C_bin(idx).ones();
        arma::uvec trivial_idx = arma::find(arma::sum(C_bin) < min_cells);
        pruned(trivial_idx).ones();

        stdout_printf("\tTrivial archetypes: %d\n", trivial_idx.n_elem);
        FLUSH;

        arma::uvec selected_archs = arma::find(pruned == 0);
        results.selected_archs = selected_archs;
        results.C_stacked = C_stacked.cols(selected_archs);
        results.H_stacked = H_stacked.rows(selected_archs);

        return (results);
    }

    unification_results unify_archetypes(arma::mat &S_r, arma::mat &C_stacked, arma::mat &H_stacked,
                                         double backbone_density, double resolution, int min_cluster_size,
                                         int thread_no, int normalization) {

        if (thread_no <= 0) {
            thread_no = SYS_THREADS_DEF;
        }

        stdout_printf("Unifying %d archetypes (%d threads):\n", C_stacked.n_cols, thread_no);
        FLUSH;

        unification_results output;

        H_stacked = arma::normalise(H_stacked, 1, 0);
        arma::sp_mat H_stacked_sp = arma::sp_mat(H_stacked);
        arma::mat H_arch = spmat_mat_product_parallel(H_stacked_sp, C_stacked, thread_no);
        H_arch.replace(arma::datum::nan, 0); // replace each NaN with 0

        SPA_results SPA_out = run_SPA(H_arch, H_arch.n_cols);
        arma::uvec candidates = SPA_out.selected_columns;
        arma::vec scores = SPA_out.column_norms;
        double x1 = arma::sum(scores);
        double x2 = arma::sum(arma::square(scores));
        double arch_no = std::round((x1 * x1) / x2);
        candidates = candidates(arma::span(0, arch_no - 1));

        stdout_printf("# unified archetypes: %d\n", (int) arch_no);
        FLUSH;

        output.selected_archetypes = candidates;

        arma::mat C_unified = C_stacked.cols(candidates);

        arma::mat X_r = normalize_mat(S_r, normalization, 0);

        arma::mat W_r_unified = X_r * C_unified;

        arma::mat H_unified = run_simplex_regression(W_r_unified, X_r, false);

        arma::uvec assigned_archetypes = arma::trans(arma::index_max(H_unified, 0));

        output.C_unified = C_unified;
        output.H_unified = H_unified;
        output.assigned_archetypes = assigned_archetypes;

        return (output);
    }

} // namespace actionet
