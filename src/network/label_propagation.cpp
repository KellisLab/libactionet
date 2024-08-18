#include "network/label_propagation.hpp"
#include "tools/enrichment.hpp"
#include "tools/normalization.hpp"
#include "utils_internal/utils_misc.hpp"

namespace actionet {
    arma::vec LPA(arma::sp_mat &G, arma::vec labels, double lambda, int iters, double sig_threshold,
                  arma::uvec fixed_labels, int thread_no) {
        int n = G.n_rows;

        arma::vec updated_labels = labels;

        arma::sp_mat H = G;
        H.diag().ones();
        H.diag() *= lambda; // add "inertia"
        H = n * normalize_adj(H, 1); // row-normalize to n

        int it;
        for (it = 0; it < iters; it++) {
            stdout_printf("\rLPA iteration %d/%d", it, iters);
            FLUSH;

            arma::vec vals = arma::unique(updated_labels);
            arma::uvec idx = arma::find(0 <= vals);
            vals = vals(idx);

            arma::mat M = one_hot_encoding(updated_labels);

            arma::mat logPvals = assess_label_enrichment(H, M, thread_no);

            arma::vec max_sig = arma::max(logPvals, 1);
            arma::vec new_labels = vals(arma::index_max(logPvals, 1));

            // Only update vertices with significant enrichment in their neighborhood
            arma::uvec sig_idx = arma::find(sig_threshold < max_sig);
            updated_labels(sig_idx) = new_labels(sig_idx);

            // revert-back the vertices that need to be fixed
            updated_labels(fixed_labels) = labels(fixed_labels);
        }
        stdout_printf("\rLPA iteration %d/%d\n", it, iters);
        FLUSH;

        return (updated_labels);
    }
} // namespace actionet
