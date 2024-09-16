#include "action/action_main.hpp"
#include "action/action_decomp.hpp"
#include "action/action_post.hpp"

namespace actionet {
    arma::field<arma::mat> runACTION(arma::mat& S_r, int k_min, int k_max, int max_it, double tol, double spec_th,
                                     int min_obs, int norm, int thread_no) {
        ResACTION trace = decompACTION(S_r, k_min, k_max, max_it, tol, thread_no);

        ResCollectArch pruned = collectArchetypes(trace.C, trace.H, spec_th, min_obs);
        ResMergeArch merged = mergeArchetypes(S_r, pruned.C_stacked, pruned.H_stacked, norm, thread_no);

        arma::field<arma::mat> out(5);
        out(0) = pruned.H_stacked;
        out(1) = pruned.C_stacked;
        out(2) = merged.H_merged;
        out(3) = merged.C_merged;
        out(4) = arma::conv_to<arma::mat>::from(merged.assigned_archetypes);

        return (out);
    }
} // namespace actionet
