#include "tools/enrichment.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"
#include <stdexcept>

namespace actionet {
    arma::mat computeGraphLabelEnrichment(const arma::sp_mat& G, const arma::mat& scores, int thread_no) {
        const arma::uword n = G.n_rows;

        // -----------------------------------------------------------
        // Single NNZ pass: compute row_sum, row_sum_sq, row_max from
        // G's CSC storage.  Avoids materializing arma::square(G) and
        // fuses three separate scans into one.
        // -----------------------------------------------------------
        arma::vec row_sum(n, arma::fill::zeros);
        arma::vec row_sum_sq(n, arma::fill::zeros);
        arma::vec row_max(n, arma::fill::zeros);

        for (arma::uword col = 0; col < G.n_cols; ++col) {
            for (arma::sp_mat::const_col_iterator it = G.begin_col(col);
                 it != G.end_col(col); ++it) {
                const arma::uword row = it.row();
                const double val = (*it);
                row_sum(row) += val;
                row_sum_sq(row) += val * val;
                if (val > row_max(row)) {
                    row_max(row) = val;
                }
            }
        }

        // -----------------------------------------------------------
        // Obs = G @ scores   (parallel SpMV)
        // -----------------------------------------------------------
        arma::mat Obs = spmat_mat_product_parallel(G, scores, thread_no);

        // -----------------------------------------------------------
        // Bennett concentration inequality with fused element-wise ops.
        //
        //   p          = mean(scores, 0)          (1 x n_labels)
        //   Exp_ij     = row_sum(i) * p(j)
        //   Lambda_ij  = Obs_ij - Exp_ij
        //   Nu_ij      = row_sum_sq(i) * p(j)
        //   scale_ij   = row_max(i) / 3
        //
        //   logPval_ij = Lambda^2 / (2 * (Nu + Lambda * scale))
        //                 where Lambda > 0, else 0
        //
        // We compute this in-place over Obs to avoid separate Lambda,
        // Nu, Lambda_scaled, and logPvals_upper matrices.
        // -----------------------------------------------------------
        const arma::rowvec p = arma::mean(scores, 0);
        const arma::uword n_labels = scores.n_cols;

        int threads_use = get_num_threads(n_labels, thread_no);
        #pragma omp parallel for num_threads(threads_use)
        for (arma::uword j = 0; j < n_labels; ++j) {
            const double pj = p(j);
            for (arma::uword i = 0; i < n; ++i) {
                double lambda = Obs(i, j) - row_sum(i) * pj;
                if (lambda <= 0.0) {
                    Obs(i, j) = 0.0;
                } else {
                    double nu = row_sum_sq(i) * pj;
                    double denom = 2.0 * (nu + lambda * (row_max(i) / 3.0));
                    Obs(i, j) = (denom > 0.0) ? (lambda * lambda) / denom : 0.0;
                }
            }
        }

        return Obs;
    }

    // ---------------------------------------------------------------
    // assess_enrichment
    //
    // For each (scores column j, association column k), compute the
    // maximum Bennett-inequality log-p-value achieved by sliding a
    // cutoff along the descending-rank order of scores.col(j) and
    // restricting the "observed" mass to features flagged in
    // associations.col(k).
    //
    //   sorted_scores = sort(scores, "descend")             (n_features x n_cols)
    //   For each association column k with n_k > 1 nonzeros:
    //     For each score column j:
    //       Take the rows of associations.col(k), map them to their
    //       positions in the descending order of scores.col(j) via
    //       perm[.] = stable_sort_index(stable_sort_index(scores.col(j), "descend"))
    //       and sort those positions ascending (`sorted_rows`).
    //       Then, letting `ii = sorted_rows[idx]`:
    //         Lambda_idx = cumsum(sorted_scores(ii, j))_over_idx
    //                    - Acumsum(ii, j) * p_k
    //         Nu_idx     = A2cumsum(ii, j) * p_k
    //         aLambda    = Lambda * a_max(j)
    //         logP_idx   = Lambda^2 / (2 * (Nu + aLambda / 3))    if Lambda > 0
    //       logPvals(k, j) = max_idx logP_idx
    //       peak_rank_idx(k, j) = argmax_idx logP_idx (in `sorted_rows` space)
    //
    // The `associations` matrix is treated as boolean (its nonzero
    // pattern), following the R/Python callers that pass either binary
    // marker matrices or CSR patterns. A local copy is produced via
    // `spones` so the caller's argument is left unmodified.
    // ---------------------------------------------------------------
    arma::field<arma::mat> assess_enrichment(const arma::mat& scores, const arma::sp_mat& associations, int thread_no) {
        if (scores.n_rows != associations.n_rows) {
            throw std::invalid_argument(
                "assess_enrichment: scores.n_rows must equal associations.n_rows (both must match number of features)");
        }

        // Local binarized copy — caller's argument is not modified.
        arma::sp_mat A = arma::spones(associations);

        arma::mat sorted_scores = arma::sort(scores, "descend");
        arma::vec a_max = arma::trans(sorted_scores.row(0));
        arma::umat perms(arma::size(scores));
        for (arma::uword j = 0; j < scores.n_cols; j++) {
            perms.col(j) =
                arma::stable_sort_index(arma::stable_sort_index(scores.col(j), "descend"));
        }

        arma::vec n_success = arma::vec(arma::trans(arma::sum(A, 0)));
        arma::vec p_success = n_success / (double)A.n_rows;

        arma::mat Acumsum = arma::cumsum(sorted_scores);
        arma::mat A2cumsum = arma::cumsum(arma::square(sorted_scores));

        const arma::uword n_conds = scores.n_cols;
        arma::mat logPvals = arma::zeros(A.n_cols, n_conds);
        arma::mat peak_rank_idx = arma::zeros(A.n_cols, n_conds);

        int threads_use = get_num_threads(A.n_cols, thread_no);

        // Restructured from the original (which allocated four n_k x n_conds
        // matrices per k iteration): process each (k, j) pair with per-j
        // scratch vectors of length n_k. This drops per-iteration allocations
        // by 4x on average and keeps peak memory O(max_n_k) per thread.
        #pragma omp parallel for num_threads(threads_use)
        for (arma::uword k = 0; k < A.n_cols; k++) {
            const int n_k = (int)n_success(k);
            if (n_k <= 1) continue;

            const double p_k = p_success(k);

            // Collect the (arbitrary-order) row indices of association column k
            // once per k; the per-j permutation and sort happens inside.
            arma::uvec assoc_rows(n_k);
            {
                arma::sp_mat::const_col_iterator it = A.begin_col(k);
                arma::sp_mat::const_col_iterator it_end = A.end_col(k);
                for (int idx = 0; it != it_end; ++it, idx++) {
                    assoc_rows[idx] = it.row();
                }
            }

            for (arma::uword j = 0; j < n_conds; j++) {
                // Map association rows to positions in descending order of
                // scores.col(j), then sort ascending along that order.
                arma::uvec sorted_rows(n_k);
                for (int idx = 0; idx < n_k; idx++) {
                    sorted_rows[idx] = perms(assoc_rows[idx], j);
                }
                sorted_rows = arma::sort(sorted_rows);

                double best_logp = 0.0;
                arma::uword best_pos = 0;
                double O_cum = 0.0;
                for (int idx = 0; idx < n_k; idx++) {
                    const arma::uword ii = sorted_rows(idx);
                    O_cum += sorted_scores(ii, j);
                    const double lambda = O_cum - Acumsum(ii, j) * p_k;
                    if (lambda <= 0.0) continue;
                    const double nu = A2cumsum(ii, j) * p_k;
                    const double denom = 2.0 * (nu + (lambda * a_max(j)) / 3.0);
                    if (denom <= 0.0) continue;
                    const double logp = (lambda * lambda) / denom;
                    if (std::isfinite(logp) && logp > best_logp) {
                        best_logp = logp;
                        best_pos = ii;
                    }
                }

                logPvals(k, j) = best_logp;
                // `peak_rank_idx` is the 0-based position (in the descending
                // sort of scores.col(j)) at which the log-p-value peaks.
                // Not a score threshold; convert via `sorted_scores(idx, j)`
                // if the peak score is desired.
                peak_rank_idx(k, j) = (double)best_pos;
            }
        }

        arma::field<arma::mat> output(2);
        output(0) = logPvals;
        output(1) = peak_rank_idx;

        return output;
    }
} // namespace actionet
