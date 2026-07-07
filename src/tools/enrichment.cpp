#include "tools/enrichment.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"

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

    arma::field<arma::mat> assess_enrichment(const arma::mat& scores, arma::sp_mat& associations, int thread_no) {
        arma::field<arma::mat> res(3);

        if (scores.n_rows != associations.n_rows) {
            stderr_printf(
                "Number of rows in scores and association matrices should both match the number of features\n");
            FLUSH;
            return (res);
        }

        associations = arma::spones(associations);

        arma::mat sorted_scores = arma::sort(scores, "descend");
        arma::vec a_max = arma::trans(sorted_scores.row(0));
        arma::umat perms(arma::size(scores));
        for (int j = 0; j < scores.n_cols; j++) {
            perms.col(j) =
                stable_sort_index(stable_sort_index(scores.col(j), "descend"));
        }

        arma::vec n_success = arma::vec(arma::trans(arma::sum(associations, 0)));
        arma::vec p_success = n_success / (double)associations.n_rows;

        arma::mat Acumsum = arma::cumsum(sorted_scores);
        arma::mat A2cumsum = arma::cumsum(arma::square(sorted_scores));

        arma::mat logPvals = arma::zeros(associations.n_cols, scores.n_cols);
        arma::mat thresholds = arma::zeros(associations.n_cols, scores.n_cols);

        int threads_use = get_num_threads(associations.n_cols, thread_no);
        #pragma omp parallel for num_threads(threads_use)
        for (size_t k = 0; k < associations.n_cols; k++) {
            int n_k = n_success(k);
            if (n_k > 1) {
                double p_k = p_success(k);

                arma::mat O = arma::zeros(n_k, scores.n_cols);
                arma::mat E = arma::zeros(n_k, scores.n_cols);
                arma::mat Nu = arma::zeros(n_k, scores.n_cols);
                arma::mat rows = arma::zeros(n_k, scores.n_cols);

                for (int j = 0; j < scores.n_cols; j++) {
                    arma::uvec perm = perms.col(j);

                    arma::uvec sorted_rows(n_k);
                    arma::sp_mat::const_col_iterator it = associations.begin_col(k);
                    arma::sp_mat::const_col_iterator it_end = associations.end_col(k);
                    for (int idx = 0; it != it_end; ++it, idx++) {
                        sorted_rows[idx] = perm[it.row()];
                    }
                    sorted_rows = arma::sort(sorted_rows);

                    for (int idx = 0; idx < n_k; idx++) {
                        int ii = sorted_rows(idx);

                        O(idx, j) = sorted_scores(ii, j);
                        E(idx, j) = Acumsum(ii, j) * p_k;
                        Nu(idx, j) = A2cumsum(ii, j) * p_k;
                        rows(idx, j) = ii;
                    }
                }
                O = arma::cumsum(O);

                arma::mat Lambda = O - E;
                arma::mat aLambda = Lambda;
                for (int j = 0; j < aLambda.n_cols; j++) {
                    aLambda.col(j) *= a_max(j);
                }

                arma::mat logPvals_k = arma::square(Lambda) / (2.0 * (Nu + (aLambda / 3.0)));
                arma::uvec idx = arma::find(Lambda <= 0);
                logPvals_k(idx) = arma::zeros(idx.n_elem);
                logPvals_k.replace(arma::datum::nan, 0);
                for (int j = 0; j < logPvals_k.n_cols; j++) {
                    arma::vec v = logPvals_k.col(j);
                    logPvals(k, j) = arma::max(v);
                    thresholds(k, j) = rows(v.index_max(), j);
                }
            }
        }

        arma::field<arma::mat> output(2);
        output(0) = logPvals;
        output(1) = thresholds;

        return (output);
    }
} // namespace actionet
