#include "action/action_main.hpp"
#include "action/action_decomp.hpp"
#include "action/action_post.hpp"

namespace actionet {
    arma::field<arma::mat> runACTION(const arma::mat& S_r, int k_min, int k_max, int max_it, double tol, double spec_th,
                                     int min_obs, int thread_no) {
        // Public contract (Plan 02): S_r is cells x k.
        // Internal pipeline (SPA, AA, simplex regression) is column-oriented and
        // expects k x cells.  Transpose at the boundary; cost is O(n_cells * k),
        // negligible compared to AA iterations.
        arma::mat S_r_internal = S_r.t();   // k x cells for internal pipeline

        ResACTION trace = decompACTION(S_r_internal, k_min, k_max, max_it, tol, thread_no);

        ResCollectArch pruned = collectArchetypes(trace.C, trace.H, spec_th, min_obs);
        // Free the per-k C/H fields from decompACTION immediately after stacking.
        // Each field entry is a (k x n_cells) or (n_cells x k) dense matrix;
        // together they occupy the same memory as C_stacked + H_stacked.
        // Releasing them before mergeArchetypes halves the peak RSS at the merge step.
        trace.C.reset();
        trace.H.reset();

        ResMergeArch merged = mergeArchetypes(S_r_internal, pruned.C_stacked, pruned.H_stacked, thread_no);
        // mergeArchetypes is done with C_stacked and H_stacked; free them now so
        // the transposition below doesn't peak alongside the merged outputs.
        arma::mat H_stacked_t = pruned.H_stacked.t();  // cells x archetypes
        pruned.H_stacked.reset();
        arma::mat C_stacked_keep = std::move(pruned.C_stacked);  // cells x archetypes (zero-copy)

        arma::field<arma::mat> out(5);
        out(0) = std::move(H_stacked_t);           // cells x archetypes
        out(1) = std::move(C_stacked_keep);        // cells x archetypes
        out(2) = merged.H_merged.t();              // cells x archetypes
        out(3) = merged.C_merged;                  // cells x archetypes
        out(4) = arma::conv_to<arma::mat>::from(merged.assigned_archetypes);

        return (out);
    }
} // namespace actionet
