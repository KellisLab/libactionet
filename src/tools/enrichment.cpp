#include "tools/enrichment.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"

namespace actionet {

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

    arma::field<arma::mat> assess_enrichment(arma::mat &scores, arma::sp_mat &associations, int thread_no) {
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
        arma::vec p_success = n_success / (double) associations.n_rows;

        arma::mat Acumsum = arma::cumsum(sorted_scores);
        arma::mat A2cumsum = arma::cumsum(arma::square(sorted_scores));

        arma::mat logPvals = arma::zeros(associations.n_cols, scores.n_cols);
        arma::mat thresholds = arma::zeros(associations.n_cols, scores.n_cols);

        // for(int k = 0; k < associations.n_cols; k++) {
        mini_thread::parallelFor(
                0, associations.n_cols, [&](size_t k) {
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
                            thresholds(k, j) = rows[v.index_max(), j];
                        }
                    }
                },
                thread_no);

        arma::field<arma::mat> output(2);
        output(0) = logPvals;
        output(1) = thresholds;

        return (output);
    }

} // namespace actionet