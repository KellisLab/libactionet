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
        ResMergeArch merged = mergeArchetypes(S_r_internal, pruned.C_stacked, pruned.H_stacked, thread_no);

        // H matrices from internal pipeline are (archetypes x cells).
        // New public contract: H is (cells x archetypes).  Transpose on output.
        // C matrices are (cells x archetypes) in both old and new contracts — unchanged.
        arma::field<arma::mat> out(5);
        out(0) = pruned.H_stacked.t();     // cells x archetypes (was archetypes x cells)
        out(1) = pruned.C_stacked;         // cells x archetypes (unchanged)
        out(2) = merged.H_merged.t();      // cells x archetypes (was archetypes x cells)
        out(3) = merged.C_merged;          // cells x archetypes (unchanged)
        out(4) = arma::conv_to<arma::mat>::from(merged.assigned_archetypes);

        return (out);
    }
} // namespace actionet
