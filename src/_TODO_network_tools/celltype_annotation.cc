#include <ACTIONet.h>

namespace ACTIONet {

    sp_mat normalize_expression_profile(sp_mat &S, int normalization = 1) {
        sp_mat T;
        if (normalization == 0) {
            // No normalization
            T = S;
        } else if (normalization == 1) {
            // LSI normalization
            T = LSI(S);
        }

        return (T);
    }

    sp_mat LSI(sp_mat &S, double size_factor = 100000) {
        sp_mat X = S;

        vec col_sum_vec = zeros(X.n_cols);
        vec row_sum_vec = zeros(X.n_rows);

        sp_mat::iterator it = X.begin();
        sp_mat::iterator it_end = X.end();
        for (; it != it_end; ++it) {
            col_sum_vec(it.col()) += (*it);
            row_sum_vec(it.row()) += (*it);
        }

        vec kappa = size_factor / col_sum_vec;
        vec IDF = log(1 + (X.n_cols / row_sum_vec));

        for (it = X.begin(); it != X.end(); ++it) {
            double x = (*it) * kappa(it.col());
            x = log(1 + x) * IDF(it.row());
            *it = x;
        }

        return (X);
    }

    mat compute_marker_aggregate_stats(sp_mat &G, sp_mat &S, sp_mat &marker_mat,
                                       double alpha = 0.85, int max_it = 5,
                                       int thread_no = 0, bool ignore_baseline_expression = false) {
        mat stats = zeros(S.n_cols, marker_mat.n_cols);

        int n = G.n_rows;
        sp_mat o = sp_mat(ones(n, 1));
        vec pr = compute_network_diffusion_fast(G, o, thread_no, alpha, max_it).col(0);

        for (int i = 0; i < marker_mat.n_cols; i++) {
            int marker_count = (int) sum(sum(spones(marker_mat.col(i))));

            int idx = 0;
            vec w = zeros(marker_count);
            vec baseline = zeros(marker_count);
            sp_mat raw_expression(S.n_cols, marker_count);
            for (sp_mat::col_iterator it = marker_mat.begin_col(i);
                 it != marker_mat.end_col(i); it++) {
                raw_expression.col(idx) = trans(S.row(it.row()));
                w(idx) = (*it);
                baseline(idx) = accu(raw_expression.col(idx));
                idx++;
            }
            if (!ignore_baseline_expression) {
                baseline = baseline / sum(baseline);
                w = w % baseline;
            }
            w = w / sqrt(sum(square(w)));

            mat imputed_expression = compute_network_diffusion_fast(
                    G, raw_expression, thread_no, alpha, max_it);

            for (int j = 0; j < imputed_expression.n_cols; j++) {
                vec ppr = imputed_expression.col(j);
                vec scores = log2(ppr / pr);
                uvec zero_idx = find(ppr == 0);
                scores(zero_idx).zeros();
                scores = scores % ppr;

                stats.col(i) += w(j) * scores;
            }
        }

        return (stats);
    }

    mat aggregate_genesets(sp_mat &G, sp_mat &S, sp_mat &marker_mat, int network_normalization_method,
                           int expression_normalization_method, int gene_scaling_method, double diffusion_alpha,
                           int thread_no) {
        if (S.n_rows != marker_mat.n_rows) {
            stderr_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
            FLUSH;
            return (mat());
        }
        if (S.n_cols != G.n_rows) {
            stderr_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
            FLUSH;
            return (mat());
        }

        sp_mat markers_mat_bin = spones(marker_mat);
        vec marker_counts = vec(trans(sum(markers_mat_bin)));

        // 0: no normalization, 1: TF/IDF
        sp_mat T = normalize_expression_profile(S, expression_normalization_method);

        // 0: pagerank, 2: sym_pagerank
        sp_mat P = normalize_adj(G, network_normalization_method);

        mat marker_stats(T.n_cols, marker_mat.n_cols);
        for (int j = 0; j < marker_mat.n_cols; j++) {
            mat marker_expr(T.n_cols, marker_counts(j));

            int idx = 0;
            for (sp_mat::col_iterator it = marker_mat.begin_col(j); it != marker_mat.end_col(j); it++) {
                double w = (*it);
                marker_expr.col(idx) = w * vec(trans(T.row(it.row())));
                idx++;
            }

            // 0: no normalization, 1: z-score, 2: RINT, 3: robust z-score
            mat marker_expr_scaled = normalize_scores(marker_expr, gene_scaling_method, thread_no);
            mat marker_expr_imputed = compute_network_diffusion_Chebyshev(P, marker_expr_scaled, thread_no);

            mat Sigma = cov(marker_expr_imputed);
            double norm_factor = sqrt(sum(Sigma.diag()));

            vec aggr_stats = sum(marker_expr_imputed, 1); // each column is a marker gene
            aggr_stats = aggr_stats / norm_factor;
            marker_stats.col(j) = aggr_stats;
        }
        mat marker_stats_smoothed = compute_network_diffusion_Chebyshev(P, marker_stats, thread_no);

        return (marker_stats_smoothed);
    }

    mat
    aggregate_genesets_mahalanobis_2archs(sp_mat &G, sp_mat &S, sp_mat &marker_mat, int network_normalization_method,
                                          int expression_normalization_method, int gene_scaling_method,
                                          double pre_alpha, double post_alpha, int thread_no) {
        if (S.n_rows != marker_mat.n_rows) {
            stderr_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
            FLUSH;
            return (mat());
        }
        if (S.n_cols != G.n_rows) {
            stderr_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
            FLUSH;
            return (mat());
        }

        // 0: pagerank, 2: sym_pagerank
        sp_mat P;
        if (pre_alpha != 0 || post_alpha != 0) {
            P = normalize_adj(G, network_normalization_method);
        }

        // 0: no normalization, 1: TF/IDF
        mat T = mat(normalize_expression_profile(S, expression_normalization_method));

        if (pre_alpha != 0) {
            mat T_t = trans(T);
            T = compute_network_diffusion_Chebyshev(P, T_t, thread_no, pre_alpha);
            T = trans(T);
        }

        mat marker_stats(T.n_cols, marker_mat.n_cols);
        parallelFor(
                0, marker_mat.n_cols, [&](int j) {

                    vec w = vec(marker_mat.col(j));
                    uvec nnz_idx = find(w != 0);
                    if (nnz_idx.n_elem != 0) {

                        mat T_scaled = T.rows(nnz_idx);
                        //0: no normalization, 1: z-score, 2: RINT, 3: robust z-score
                        if (gene_scaling_method != 0) {
                            T_scaled = normalize_scores(T_scaled, gene_scaling_method, thread_no);
                        }
                        T_scaled = T_scaled.each_col() % w(nnz_idx);

                        uvec idx(2);
                        rowvec ss = sum(T_scaled);
                        idx(0) = index_min(ss);
                        idx(1) = index_max(ss);

                        mat W0 = T_scaled.cols(idx);

                        field <mat> AA_res = run_AA(T_scaled, W0, 100);
                        mat C = AA_res(0);
                        mat H = AA_res(1);
                        mat W = T_scaled * C;
                        uword selected_arch0 = index_min(sum(W));
                        uword selected_arch1 = index_max(sum(W));
                        vec mu = W.col(selected_arch0);

                        double p = T_scaled.n_rows;
                        double n = T_scaled.n_cols;

                        mat Delta = T_scaled.each_col() - mu;

                        mat sigma = Delta * trans(Delta) / (n - 1);
                        mat sigma_inv = pinv(sigma);

                        for (int k = 0; k < n; k++) {
                            vec delta = Delta.col(k);
                            double dist = dot(delta, sigma_inv * delta);
                            double z = (dist - p) / sqrt(2 * p);
                            z = z < 0 ? 0 : z;

                            marker_stats(k, j) = sign(mean(delta)) * z;
                        }
                    }
                },
                thread_no);

        marker_stats.replace(datum::nan, 0);

        mat marker_stats_smoothed = marker_stats; // zscore(marker_stats, thread_no);
        if (post_alpha != 0) {
            stdout_printf("Post-smoothing expression values ... ");
            marker_stats_smoothed = compute_network_diffusion_Chebyshev(P, marker_stats_smoothed, thread_no,
                                                                        post_alpha);
            stdout_printf("done\n");
            FLUSH;
        }

        return (marker_stats_smoothed);
    }

    mat aggregate_genesets_mahalanobis_2gmm(sp_mat &G, sp_mat &S, sp_mat &marker_mat, int network_normalization_method,
                                            int expression_normalization_method, int gene_scaling_method,
                                            double pre_alpha, double post_alpha, int thread_no) {
        if (S.n_rows != marker_mat.n_rows) {
            stderr_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
            FLUSH;
            return (mat());
        }
        if (S.n_cols != G.n_rows) {
            stderr_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
            FLUSH;
            return (mat());
        }

        // 0: pagerank, 2: sym_pagerank
        sp_mat P;
        if (pre_alpha != 0 || post_alpha != 0) {
            P = normalize_adj(G, network_normalization_method);
        }

        // 0: no normalization, 1: TF/IDF
        mat T = mat(normalize_expression_profile(S, expression_normalization_method));

        if (pre_alpha != 0) {
            mat T_t = trans(T);
            T = compute_network_diffusion_Chebyshev(P, T_t, thread_no, pre_alpha);
            T = trans(T);
        }

        mat marker_stats(T.n_cols, marker_mat.n_cols);
        parallelFor(
                0, marker_mat.n_cols, [&](int j) {

                    vec w = vec(marker_mat.col(j));
                    uvec nnz_idx = find(w != 0);
                    if (nnz_idx.n_elem != 0) {

                        mat T_scaled = T.rows(nnz_idx);
                        //0: no normalization, 1: z-score, 2: RINT, 3: robust z-score
                        if (gene_scaling_method != 0) {
                            T_scaled = normalize_scores(T_scaled, gene_scaling_method, thread_no);
                        }
                        T_scaled = T_scaled.each_col() % w(nnz_idx);

                        gmm_full model;

                        bool status = model.learn(T_scaled, 2, maha_dist, static_spread, 10, 10, 1e-10, false);
                        mat W = model.means;

                        uword selected_arch0 = index_min(sum(W));
                        uword selected_arch1 = index_max(sum(W));
                        vec mu = W.col(selected_arch0);

                        double p = T_scaled.n_rows;
                        double n = T_scaled.n_cols;

                        mat Delta = T_scaled.each_col() - mu;

                        //mat sigma = cov(trans(T_scaled));
                        mat sigma = Delta * trans(Delta) / (n - 1);
                        mat sigma_inv = pinv(sigma);

                        for (int k = 0; k < n; k++) {
                            vec delta = Delta.col(k);
                            double dist = dot(delta, sigma_inv * delta);
                            double z = (dist - p) / sqrt(2 * p);
                            z = z < 0 ? 0 : z;

                            marker_stats(k, j) = sign(mean(delta)) * z;
                        }
                    }
                },
                thread_no);

        marker_stats.replace(datum::nan, 0);

        mat marker_stats_smoothed = marker_stats; // zscore(marker_stats, thread_no);
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
