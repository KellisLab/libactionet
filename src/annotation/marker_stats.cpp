#include "annotation/marker_stats.hpp"
#include "action/aa.hpp"
#include "network/network_diffusion.hpp"
#include "utils_internal/utils_matrix.hpp"

double tol_approx = 1E-8;

namespace actionet {
    arma::mat computeFeatureStats(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X, int norm_method,
                                  double alpha, int max_it, bool approx, int thread_no, bool ignore_baseline) {
        arma::mat stats = arma::zeros(S.n_cols, X.n_cols);

        int n = G.n_rows;
        arma::sp_mat o = arma::sp_mat(arma::ones(n, 1));
        arma::vec pr = computeNetworkDiffusion(G, o, alpha, max_it, thread_no, approx).col(0);

        for (int i = 0; i < X.n_cols; i++) {
            // int marker_count = (int)sum(sum(spones(X.col(i))));
            int marker_count = arma::accu(arma::spones(X.col(i)));

            int idx = 0;
            arma::vec w = arma::zeros(marker_count);
            arma::vec baseline = arma::zeros(marker_count);
            arma::sp_mat raw_expression(S.n_cols, marker_count);
            for (arma::sp_mat::col_iterator it = X.begin_col(i);
                 it != X.end_col(i); ++it) {
                raw_expression.col(idx) = trans(S.row(it.row()));
                w(idx) = (*it);
                baseline(idx) = arma::accu(raw_expression.col(idx));
                idx++;
            }
            if (!ignore_baseline) {
                baseline = baseline / arma::sum(baseline);
                w = w % baseline;
            }
            w = w / std::sqrt(arma::sum(arma::square(w)));

            arma::mat imputed_expression = computeNetworkDiffusion(G, raw_expression, alpha, max_it, thread_no,
                                                                   approx, norm_method, tol_approx);

            for (int j = 0; j < imputed_expression.n_cols; j++) {
                arma::vec ppr = imputed_expression.col(j);
                arma::vec scores = arma::log2(ppr / pr);
                arma::uvec zero_idx = arma::find(ppr == 0);
                scores(zero_idx).zeros();
                scores = scores % ppr;

                stats.col(i) += w(j) * scores;
            }
        }

        return (stats);
    }

    arma::mat computeFeatureStatsVision(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X,
                                        int norm_method, double alpha, int max_it,
                                        bool approx, int thread_no) {
        // `X` is features; formerly `marker_mat`
        if (S.n_rows != X.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (S.n_rows != X.n_rows)");
        }
        if (S.n_cols != G.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (S.n_cols != G.n_rows)");
        }

        arma::sp_mat St = arma::trans(S);
        arma::mat stats = arma::mat(spmat_spmat_product(St, X));

        // Compute cell-specific stats to adjust for depth, etc.
        arma::vec mu = arma::zeros(S.n_cols);
        arma::vec nnz = arma::zeros(S.n_cols);
        for (arma::sp_mat::const_iterator it = S.begin(); it != S.end(); ++it) {
            mu[it.col()] += (*it);
            nnz[it.col()]++;
        }
        mu /= S.n_rows;
        arma::vec p_nnz = nnz / S.n_rows; // 1 - p_zero

        arma::vec sigma_sq = arma::zeros(S.n_cols);
        for (arma::sp_mat::const_iterator it = S.begin(); it != S.end(); ++it) {
            float delta = mu[it.col()] - (*it);
            sigma_sq[it.col()] += delta * delta;
        }
        sigma_sq += (S.n_rows * (1 - p_nnz)) % arma::square(mu); // Adjust for nnzs
        sigma_sq /= (S.n_rows - 1);

        // Standardize using sampling mean (from Vision: https://www.nature.com/articles/s41467-019-12235-0)
        arma::rowvec k1 = arma::rowvec(arma::sum(X));
        arma::rowvec k2 = arma::rowvec(arma::sum(arma::square(X)));

        arma::mat sampling_mu = mu * k1;
        arma::mat sampling_sigma_sq = sigma_sq * k2;
        arma::mat marker_stats = (stats - sampling_mu) / arma::sqrt(sampling_sigma_sq);
        marker_stats.replace(arma::datum::nan, 0);

        arma::mat marker_stats_smoothed = marker_stats;
        if (alpha != 0) {
            // marker_stats_smoothed = actionet::computeNetworkDiffusion(G, marker_stats_smoothed, alpha, max_it, thread_no, true, norm_method, 1E-8);
            marker_stats = actionet::computeNetworkDiffusion(G, marker_stats_smoothed, alpha, max_it, thread_no, approx,
                                                             norm_method, tol_approx);
        }

        // arma::field<arma::mat> out(3);
        // out(0) = marker_stats_smoothed;
        // out(1) = marker_stats;
        // out(2) = stats;

        return (marker_stats);
    }
} // namespace actionet
