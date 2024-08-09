#include "label_propagation.hpp"

namespace ACTIONet {

    arma::mat assess_label_enrichment(arma::sp_mat &H, arma::mat &M, int thread_no) {
        arma::mat Obs = spmat_mat_product_parallel(H, M, thread_no);

        arma::rowvec p = mean(M, 0);
        arma::mat Exp = sum(H, 1) * p;

        arma::mat Lambda = Obs - Exp;

        arma::mat Nu = arma::sum(arma::square(H), 1) * p;
        arma::vec a = arma::vec(arma::max(H, 1));

        arma::mat Lambda_scaled = Lambda;
        for (int j = 0; j < Lambda_scaled.n_rows; j++) {
            Lambda_scaled.row(j) *= (a(j) / 3);
        }

        arma::mat logPvals_upper = arma::square(Lambda) / (2 * (Nu + Lambda_scaled));
        arma::uvec lidx = arma::find(Lambda <= 0);
        logPvals_upper(lidx) = arma::zeros(lidx.n_elem);
        logPvals_upper.replace(arma::datum::nan, 0); // replace each NaN with 0

        return logPvals_upper;
    }


    arma::vec LPA(arma::sp_mat &G, arma::vec labels, double lambda, int iters, double sig_threshold,
                  arma::uvec fixed_labels, int thread_no) {

        int n = G.n_rows;

        arma::vec updated_labels = labels;

        arma::sp_mat H = G;
        H.diag().ones();
        H.diag() *= lambda;          // add "inertia"
        H = n * normalize_adj(H, 1); // row-normalize to n

        int it = 0;
        for (it; it < iters; it++) {
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
        stdout_printf("\r\tLPA iteration %d/%d", it, iters);
        FLUSH;

        return (updated_labels);
    }

} // namespace ACTIONet
