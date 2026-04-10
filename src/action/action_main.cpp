#include "action/action_main.hpp"
#include "action/action_decomp.hpp"
#include "action/action_post.hpp"

namespace actionet {
    arma::field<arma::mat> runACTION(const arma::mat& S_r, int k_min, int k_max, int max_it, double tol, double spec_th,
                                     int min_obs, int thread_no, bool return_c_matrices) {
        // Public contract (Plan 02): S_r is cells x k.
        // Internal pipeline (SPA, AA, simplex regression) is column-oriented and
        // expects k x cells.  Transpose at the boundary; cost is O(n_cells * k),
        // negligible compared to AA iterations.
        arma::mat S_r_internal = S_r.t();   // k x cells for internal pipeline

        ResACTION trace = decompACTION(S_r_internal, k_min, k_max, max_it, tol, thread_no);

        arma::mat C_stacked = std::move(trace.C_stacked);
        arma::mat H_stacked = std::move(trace.H_stacked);
        trace.selected_cols.reset();

        ResCollectArch pruned = collectArchetypes(C_stacked, H_stacked, spec_th, min_obs);
        // collectArchetypes is done with the full stacked buffers; free them now so
        // mergeArchetypes doesn't peak alongside both the full-T and retained-R copies.
        C_stacked.reset();
        H_stacked.reset();

        ResMergeArch merged = mergeArchetypes(S_r_internal, pruned.C_stacked, pruned.H_stacked, thread_no);
        if (!return_c_matrices) {
            // C matrices are no longer needed after merge. Release them before
            // H transpositions to reduce peak RSS.
            pruned.C_stacked.reset();
            merged.C_merged.reset();
        }
        // mergeArchetypes is done with C_stacked and H_stacked; free them now so
        // the transposition below doesn't peak alongside the merged outputs.
        arma::mat H_stacked_t = pruned.H_stacked.t();  // cells x archetypes
        pruned.H_stacked.reset();
        arma::mat C_stacked_keep;
        if (return_c_matrices) {
            C_stacked_keep = std::move(pruned.C_stacked);  // cells x archetypes (zero-copy)
        }
        arma::mat H_merged_t = merged.H_merged.t();    // cells x archetypes
        merged.H_merged.reset();
        arma::mat C_merged_keep;
        if (return_c_matrices) {
            C_merged_keep = std::move(merged.C_merged);    // cells x archetypes
        }

        arma::field<arma::mat> out(5);
        out(0) = std::move(H_stacked_t);           // cells x archetypes
        out(1) = std::move(C_stacked_keep);        // cells x archetypes
        out(2) = std::move(H_merged_t);            // cells x archetypes
        out(3) = std::move(C_merged_keep);         // cells x archetypes
        out(4) = arma::conv_to<arma::mat>::from(merged.assigned_archetypes);

        return (out);
    }
} // namespace actionet
