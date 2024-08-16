#include "actionet/annotation.hpp"
#include "action/aa.hpp"
#include "actionet/network_diffusion.hpp"
#include "tools/normalization.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"

// TODO: Remove and replace with generic mat normalization functions
arma::sp_mat normalize_expression_profile(arma::sp_mat &S, int normalization) {
    arma::sp_mat T;
    if (normalization == 0) {
        // No normalization
        T = S;
    } else if (normalization == 1) {
        // LSI normalization
        T = ACTIONet::LSI(S);
    }

    return (T);
}

namespace ACTIONet {

    arma::mat compute_marker_aggregate_stats(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat, double alpha,
                                             int max_it, int thread_no, bool ignore_baseline_expression) {
        arma::mat stats = arma::zeros(S.n_cols, marker_mat.n_cols);

        int n = G.n_rows;
        arma::sp_mat o = arma::sp_mat(arma::ones(n, 1));
        arma::vec pr = compute_network_diffusion_fast(G, o, thread_no, alpha, max_it).col(0);

        for (int i = 0; i < marker_mat.n_cols; i++) {
            int marker_count = (int) sum(sum(spones(marker_mat.col(i))));

            int idx = 0;
            arma::vec w = arma::zeros(marker_count);
            arma::vec baseline = arma::zeros(marker_count);
            arma::sp_mat raw_expression(S.n_cols, marker_count);
            for (arma::sp_mat::col_iterator it = marker_mat.begin_col(i);
                 it != marker_mat.end_col(i); it++) {
                raw_expression.col(idx) = trans(S.row(it.row()));
                w(idx) = (*it);
                baseline(idx) = arma::accu(raw_expression.col(idx));
                idx++;
            }
            if (!ignore_baseline_expression) {
                baseline = baseline / arma::sum(baseline);
                w = w % baseline;
            }
            w = w / std::sqrt(arma::sum(arma::square(w)));

            arma::mat imputed_expression = compute_network_diffusion_fast(
                    G, raw_expression, thread_no, alpha, max_it);

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

    arma::mat aggregate_genesets(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                 int network_normalization_method, int expression_normalization_method,
                                 int gene_scaling_method, double diffusion_alpha, int thread_no) {

        if (S.n_rows != marker_mat.n_rows) {
            stderr_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
            FLUSH;
            return (arma::mat());
        }
        if (S.n_cols != G.n_rows) {
            stderr_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
            FLUSH;
            return (arma::mat());
        }

        arma::sp_mat markers_mat_bin = arma::spones(marker_mat);
        arma::vec marker_counts = arma::vec(arma::trans(arma::sum(markers_mat_bin)));

        // 0: no normalization, 1: TF/IDF
        arma::sp_mat T = normalize_expression_profile(S, expression_normalization_method);

        // 0: pagerank, 2: sym_pagerank
        arma::sp_mat P = normalize_adj(G, network_normalization_method);

        arma::mat marker_stats(T.n_cols, marker_mat.n_cols);
        for (int j = 0; j < marker_mat.n_cols; j++) {
            arma::mat marker_expr(T.n_cols, marker_counts(j));

            int idx = 0;
            for (arma::sp_mat::col_iterator it = marker_mat.begin_col(j); it != marker_mat.end_col(j); it++) {
                double w = (*it);
                marker_expr.col(idx) = w * arma::vec(arma::trans(T.row(it.row())));
                idx++;
            }

            // 0: no normalization, 1: z-score, 2: RINT, 3: robust z-score
            arma::mat marker_expr_scaled = normalize_scores(marker_expr, gene_scaling_method, thread_no);
            arma::mat marker_expr_imputed = compute_network_diffusion_Chebyshev(P, marker_expr_scaled, thread_no);

            arma::mat Sigma = arma::cov(marker_expr_imputed);
            double norm_factor = std::sqrt(arma::sum(Sigma.diag()));

            arma::vec aggr_stats = arma::sum(marker_expr_imputed, 1); // each column is a marker gene
            aggr_stats = aggr_stats / norm_factor;
            marker_stats.col(j) = aggr_stats;
        }
        arma::mat marker_stats_smoothed = compute_network_diffusion_Chebyshev(P, marker_stats, thread_no);

        return (marker_stats_smoothed);
    }

    arma::field<arma::mat> aggregate_genesets_vision(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                     int network_normalization_method, double alpha, int thread_no) {
        arma::field<arma::mat> out(3);

        if (S.n_rows != marker_mat.n_rows) {
            stderr_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
            FLUSH;
            return (out);
        }
        if (S.n_cols != G.n_rows) {
            stderr_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
            FLUSH;
            return (out);
        }

        // 0: pagerank, 2: sym_pagerank
        arma::sp_mat P;
        if (alpha != 0) {
            P = normalize_adj(G, network_normalization_method);
        }

        arma::mat X = arma::mat(marker_mat);
        arma::sp_mat St = arma::trans(S);

        arma::mat stats = spmat_mat_product_parallel(St, X, thread_no);

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
        arma::rowvec k1 = arma::rowvec(arma::sum(marker_mat));
        arma::rowvec k2 = arma::rowvec(arma::sum(arma::square(marker_mat)));

        arma::mat sampling_mu = mu * k1;
        arma::mat sampling_sigma_sq = sigma_sq * k2;
        arma::mat marker_stats = (stats - sampling_mu) / arma::sqrt(sampling_sigma_sq);
        marker_stats.replace(arma::datum::nan, 0);

        arma::mat marker_stats_smoothed = marker_stats;
        if (alpha != 0) {
            stdout_printf("Smoothing geneset scores ... ");
            marker_stats_smoothed = ACTIONet::compute_network_diffusion_Chebyshev(P, marker_stats_smoothed, thread_no,
                                                                                  alpha);
            stdout_printf("done\n");
            FLUSH;
        }

        out(0) = marker_stats_smoothed;
        out(1) = marker_stats;
        out(2) = stats;

        return (out);
    }

    arma::mat aggregate_genesets_mahalanobis_2archs(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                    int network_normalization_method,
                                                    int expression_normalization_method, int gene_scaling_method,
                                                    double pre_alpha, double post_alpha, int thread_no) {
        if (S.n_rows != marker_mat.n_rows) {
            stderr_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
            FLUSH;
            return (arma::mat());
        }
        if (S.n_cols != G.n_rows) {
            stderr_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
            FLUSH;
            return (arma::mat());
        }

        // 0: pagerank, 2: sym_pagerank
        arma::sp_mat P;
        if (pre_alpha != 0 || post_alpha != 0) {
            P = normalize_adj(G, network_normalization_method);
        }

        // 0: no normalization, 1: TF/IDF
        arma::mat T = arma::mat(normalize_expression_profile(S, expression_normalization_method));

        if (pre_alpha != 0) {
            arma::mat T_t = trans(T);
            T = compute_network_diffusion_Chebyshev(P, T_t, thread_no, pre_alpha);
            T = arma::trans(T);
        }

        arma::mat marker_stats(T.n_cols, marker_mat.n_cols);
        mini_thread::parallelFor(
                0, marker_mat.n_cols, [&](int j) {

                    arma::vec w = arma::vec(marker_mat.col(j));
                    arma::uvec nnz_idx = find(w != 0);
                    if (nnz_idx.n_elem != 0) {

                        arma::mat T_scaled = T.rows(nnz_idx);
                        //0: no normalization, 1: z-score, 2: RINT, 3: robust z-score
                        if (gene_scaling_method != 0) {
                            T_scaled = normalize_scores(T_scaled, gene_scaling_method, thread_no);
                        }
                        T_scaled = T_scaled.each_col() % w(nnz_idx);

                        arma::uvec idx(2);
                        arma::rowvec ss = arma::sum(T_scaled);
                        idx(0) = arma::index_min(ss);
                        idx(1) = arma::index_max(ss);

                        arma::mat W0 = T_scaled.cols(idx);

                        arma::field<arma::mat> AA_res = run_AA(T_scaled, W0, 100);
                        arma::mat C = AA_res(0);
                        arma::mat H = AA_res(1);
                        arma::mat W = T_scaled * C;
                        arma::uword selected_arch0 = arma::index_min(sum(W));
                        arma::uword selected_arch1 = arma::index_max(sum(W));
                        arma::vec mu = W.col(selected_arch0);

                        double p = T_scaled.n_rows;
                        double n = T_scaled.n_cols;

                        arma::mat Delta = T_scaled.each_col() - mu;

                        arma::mat sigma = Delta * arma::trans(Delta) / (n - 1);
                        arma::mat sigma_inv = arma::pinv(sigma);

                        for (int k = 0; k < n; k++) {
                            arma::vec delta = Delta.col(k);
                            double dist = dot(delta, sigma_inv * delta);
                            double z = (dist - p) / sqrt(2 * p);
                            z = z < 0 ? 0 : z;

                            marker_stats(k, j) = arma::sign(arma::mean(delta)) * z;
                        }
                    }
                },
                thread_no);

        marker_stats.replace(arma::datum::nan, 0);

        arma::mat marker_stats_smoothed = marker_stats; // zscore(marker_stats, thread_no);
        if (post_alpha != 0) {
            stdout_printf("Post-smoothing expression values ... ");
            marker_stats_smoothed = compute_network_diffusion_Chebyshev(P, marker_stats_smoothed, thread_no,
                                                                        post_alpha);
            stdout_printf("done\n");
            FLUSH;
        }

        return (marker_stats_smoothed);
    }

    arma::mat aggregate_genesets_mahalanobis_2gmm(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                  int network_normalization_method, int expression_normalization_method,
                                                  int gene_scaling_method, double pre_alpha, double post_alpha,
                                                  int thread_no) {
        if (S.n_rows != marker_mat.n_rows) {
            stderr_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
            FLUSH;
            return (arma::mat());
        }
        if (S.n_cols != G.n_rows) {
            stderr_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
            FLUSH;
            return (arma::mat());
        }

        // 0: pagerank, 2: sym_pagerank
        arma::sp_mat P;
        if (pre_alpha != 0 || post_alpha != 0) {
            P = normalize_adj(G, network_normalization_method);
        }

        // 0: no normalization, 1: TF/IDF
        arma::mat T = arma::mat(normalize_expression_profile(S, expression_normalization_method));

        if (pre_alpha != 0) {
            arma::mat T_t = arma::trans(T);
            T = compute_network_diffusion_Chebyshev(P, T_t, thread_no, pre_alpha);
            T = arma::trans(T);
        }

        arma::mat marker_stats(T.n_cols, marker_mat.n_cols);
        mini_thread::parallelFor(
                0, marker_mat.n_cols, [&](int j) {

                    arma::vec w = arma::vec(marker_mat.col(j));
                    arma::uvec nnz_idx = arma::find(w != 0);
                    if (nnz_idx.n_elem != 0) {

                        arma::mat T_scaled = T.rows(nnz_idx);
                        //0: no normalization, 1: z-score, 2: RINT, 3: robust z-score
                        if (gene_scaling_method != 0) {
                            T_scaled = normalize_scores(T_scaled, gene_scaling_method, thread_no);
                        }
                        T_scaled = T_scaled.each_col() % w(nnz_idx);

                        arma::gmm_full model;

                        bool status = model.learn(T_scaled, 2, arma::maha_dist,
                                                  arma::static_spread, 10, 10, 1e-10, false);
                        arma::mat W = model.means;

                        arma::uword selected_arch0 = arma::index_min(arma::sum(W));
                        arma::uword selected_arch1 = arma::index_max(arma::sum(W));
                        arma::vec mu = W.col(selected_arch0);

                        double p = T_scaled.n_rows;
                        double n = T_scaled.n_cols;

                        arma::mat Delta = T_scaled.each_col() - mu;

                        //mat sigma = cov(trans(T_scaled));
                        arma::mat sigma = Delta * arma::trans(Delta) / (n - 1);
                        arma::mat sigma_inv = arma::pinv(sigma);

                        for (int k = 0; k < n; k++) {
                            arma::vec delta = Delta.col(k);
                            double dist = arma::dot(delta, sigma_inv * delta);
                            double z = (dist - p) / std::sqrt(2 * p);
                            z = z < 0 ? 0 : z;

                            marker_stats(k, j) = arma::sign(arma::mean(delta)) * z;
                        }
                    }
                },
                thread_no);

        marker_stats.replace(arma::datum::nan, 0);

        arma::mat marker_stats_smoothed = marker_stats; // zscore(marker_stats, thread_no);
        if (post_alpha != 0) {
            stdout_printf("Post-smoothing expression values ... ");
            marker_stats_smoothed = compute_network_diffusion_Chebyshev(P, marker_stats_smoothed, thread_no,
                                                                        post_alpha);
            stdout_printf("done\n");
            FLUSH;
        }

        return (marker_stats_smoothed);
    }

} // namespace ACTIONet
