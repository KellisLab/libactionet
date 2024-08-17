#include "celltype_annotation_ext.h"

// NOTE: RIN_transform() removed

double F2z(double F, double d1, double d2) {
    double mu = d2 / (d2 - 2);                                                               // Only valud if d2 > 2
    double sigma_sq = (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4)); // Only valid when d2 > 4

    double z = (F - mu) / std::sqrt(sigma_sq);
    return (z);
}

arma::mat doubleNorm(arma::mat &X) {
    arma::vec rs = arma::sum(X, 1);
    arma::vec cs = arma::trans(arma::sum(X, 0));

    arma::mat Dr = arma::diagmat(1 / arma::sqrt(rs));
    arma::mat Dc = arma::diagmat(1 / arma::sqrt(cs));

    arma::mat Y = Dr * X * Dc;

    return (Y);
}

arma::sp_mat scale_expression(arma::sp_mat &S) {
    arma::sp_mat T = S;

    arma::sp_mat::iterator it = T.begin();
    arma::sp_mat::iterator it_end = T.end();

    arma::vec mu = arma::vec(arma::sum(T, 1)) / arma::vec(arma::sum(arma::spones(T), 1));
    for (; it != it_end; ++it) {
        (*it) -= mu(it.row());
    }
    arma::vec sigma = arma::vec(arma::sum(arma::square(T), 1));

    T = S;
    for (; it != it_end; ++it) {
        (*it) /= sigma(it.row());
    }

    return (T);
}

arma::mat compute_marker_aggregate_stats_basic_sum(arma::sp_mat &S, arma::sp_mat &marker_mat) {
    marker_mat = arma::normalise(marker_mat, 1, 0);
    arma::sp_mat X = arma::trans(marker_mat);

    S = scale_expression(S);
    arma::mat stats = arma::mat(trans(X * S));

    return (stats);
}

arma::mat compute_marker_aggregate_stats_basic_sum_perm(arma::sp_mat &S, arma::sp_mat &marker_mat, int perm_no,
                                                        int thread_no) {

    marker_mat = arma::normalise(marker_mat, 1, 0);
    arma::mat X = arma::trans(arma::mat(marker_mat));

    // S = scale_expression(S);
    arma::mat stats = arma::mat(arma::trans(arma::sp_mat(X * S)));

    int N = X.n_cols;

    arma::mat E = arma::zeros(arma::size(stats));
    arma::mat Esq = arma::zeros(arma::size(stats));

    mini_thread::parallelFor(
            0, perm_no, [&](size_t i) {
                arma::uvec perm = arma::randperm(N);
                arma::mat rand_stats = arma::mat(arma::trans(arma::sp_mat(X.cols(perm) * S)));
                arma::mat shifted_vals = (rand_stats - stats);
                E += shifted_vals;
                Esq += square(shifted_vals);
            },
            thread_no);
    arma::mat mu = E / perm_no + stats;
    arma::mat sigma = arma::sqrt((Esq - arma::square(E) / perm_no) / (perm_no - 1));
    arma::mat Z = (stats - mu) / sigma;

    return (Z);
}

arma::mat compute_marker_aggregate_stats_basic_sum_perm_smoothed(arma::sp_mat &G, arma::sp_mat &S,
                                                                 arma::sp_mat &marker_mat, double alpha,
                                                                 int max_it, int perm_no, int thread_no) {

    marker_mat = arma::normalise(marker_mat, 1, 0);
    arma::mat X = arma::trans(arma::mat(marker_mat));

    S = scale_expression(S);
    arma::sp_mat raw_stats = arma::trans(arma::sp_mat(X * S));
    arma::mat stats = actionet::compute_network_diffusion_fast(G, raw_stats, thread_no, alpha,
                                                               max_it); // * diagmat(vec(trans(sum(raw_stats))));

    int N = X.n_cols;

    arma::mat E = arma::zeros(arma::size(stats));
    arma::mat Esq = arma::zeros(arma::size(stats));

    mini_thread::parallelFor(
            0, perm_no, [&](size_t i) {
                arma::uvec perm = arma::randperm(N);
                arma::sp_mat raw_rand_stats = trans(arma::sp_mat(X.cols(perm) * S));
                arma::mat rand_stats = actionet::compute_network_diffusion_fast(G, raw_rand_stats, 1, alpha,
                                                                                max_it); // * diagmat(vec(trans(sum(raw_rand_stats))));

                arma::mat shifted_vals = (rand_stats - stats);
                E += shifted_vals;
                Esq += arma::square(shifted_vals);
            },
            thread_no);
    arma::mat mu = E / perm_no + stats;
    arma::mat sigma = arma::sqrt((Esq - arma::square(E) / perm_no) / (perm_no - 1));
    arma::mat Z = (stats - mu) / sigma;

    return (Z);
}

arma::mat compute_marker_aggregate_stats_basic_sum_smoothed(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                            double alpha, int max_it, int perm_no, int thread_no) {

    marker_mat = arma::normalise(marker_mat, 1, 0);
    arma::mat X = arma::trans(arma::mat(marker_mat));

    arma::sp_mat raw_stats = arma::trans(arma::sp_mat(X * S));
    arma::mat stats = actionet::compute_network_diffusion_fast(G, raw_stats, thread_no, alpha, max_it) *
                      arma::diagmat(arma::vec(arma::trans(arma::sum(raw_stats))));

    return (stats);
}

arma::mat compute_marker_aggregate_stats_basic_sum_smoothed_normalized(arma::sp_mat &G, arma::sp_mat &S,
                                                                       arma::sp_mat &marker_mat, double alpha,
                                                                       int max_it, int perm_no, int thread_no) {
    marker_mat = arma::normalise(marker_mat, 1, 0);
    arma::mat X = arma::trans(arma::mat(marker_mat));

    arma::sp_mat raw_stats = arma::trans(arma::sp_mat(X * S));
    arma::mat stats = actionet::compute_network_diffusion_fast(G, raw_stats, thread_no, alpha, max_it) *
                      arma::diagmat(arma::vec(arma::trans(arma::sum(raw_stats))));

    arma::sp_mat p = arma::trans(arma::sum(S));
    arma::vec pr = actionet::compute_network_diffusion_fast(G, p, thread_no, alpha, max_it).col(0);

    for (int j = 0; j < stats.n_cols; j++) {
        arma::vec ppr = stats.col(j);
        arma::vec scores_norm = arma::log2(ppr / pr);
        arma::uvec zero_idx = arma::find(ppr == 0);
        scores_norm(zero_idx).zeros();
        scores_norm = scores_norm % ppr;

        stats.col(j) = scores_norm;
    }

    return (stats);
}

arma::mat compute_marker_aggregate_stats_basic_sum_perm_smoothed_v2(arma::sp_mat &G, arma::sp_mat &S,
                                                                    arma::sp_mat &marker_mat, double alpha, int max_it,
                                                                    int perm_no, int thread_no) {

    marker_mat = arma::normalise(marker_mat, 1, 0);
    arma::mat X = arma::trans(arma::mat(marker_mat));

    arma::sp_mat raw_stats = arma::trans(arma::sp_mat(X * S));
    arma::mat stats = actionet::compute_network_diffusion_fast(G, raw_stats, thread_no, alpha, max_it) *
                      arma::diagmat(arma::vec(arma::trans(arma::sum(raw_stats))));

    arma::mat raw_stats_mat = arma::mat(raw_stats);

    int N = X.n_cols;
    arma::mat E = arma::zeros(arma::size(stats));
    arma::mat Esq = arma::zeros(arma::size(stats));

    mini_thread::parallelFor(
            0, perm_no, [&](size_t i) {
                arma::uvec perm = arma::randperm(N);

                arma::sp_mat raw_rand_stats = arma::sp_mat(raw_stats_mat.rows(perm));
                arma::mat rand_stats = actionet::compute_network_diffusion_fast(G, raw_rand_stats, 1, alpha, max_it) *
                                       arma::diagmat(arma::vec(arma::trans(arma::sum(raw_rand_stats))));

                E += rand_stats;
                Esq += arma::square(rand_stats);
            },
            thread_no);

    arma::mat mu = E / perm_no;
    arma::mat sigma = arma::sqrt(Esq / perm_no - arma::square(mu));
    arma::mat Z = (stats - mu) / sigma;

    return (Z);
}

arma::mat compute_marker_aggregate_stats_nonparametric(arma::mat &S, arma::sp_mat &marker_mat, int thread_no) {
    arma::mat St = arma::trans(S);

    // RIN_transform removed from package.
    arma::mat Z = RIN_transform(St, thread_no); // cell x gene

    arma::mat stats = arma::zeros(Z.n_rows, marker_mat.n_cols);
    for (int i = 0; i < marker_mat.n_cols; i++) {
        arma::vec v = arma::vec(marker_mat.col(i));
        arma::uvec idx = find(v != 0);
        arma::vec w = v(idx);
        double sigma = std::sqrt(arma::sum(arma::square(w)));
        stats.col(i) = sum(Z.cols(idx), 1) / sigma;
    }

    return (stats);
}

arma::mat compute_marker_aggregate_stats_TFIDF_sum_smoothed(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                            double alpha, int max_it, int perm_no, int thread_no,
                                                            int normalization) {
    marker_mat = arma::normalise(marker_mat, 1, 0);
    arma::mat X = arma::trans(arma::mat(marker_mat));

    arma::sp_mat T;
    if (normalization == 0) {
        T = S;
    } else if (normalization == 1) {
        T = LSI(S);
    }

    arma::vec base = arma::vec(arma::trans(T.row(0)));

    arma::sp_mat::iterator it = T.begin();
    arma::sp_mat::iterator it_end = T.end();
    arma::vec E = arma::zeros(T.n_cols);
    arma::vec Esq = arma::zeros(T.n_cols);
    for (; it != it_end; ++it) {
        double x = *it - base(it.col());
        E(it.col()) += x;
        Esq(it.col()) += (x * x);
    }
    arma::mat mu = E / T.n_rows + base;
    arma::mat sigma = arma::sqrt((Esq - square(E) / T.n_rows) / (T.n_rows - 1));

    arma::vec w1 = arma::vec(arma::trans(arma::sum(marker_mat, 0)));
    arma::vec w2 = arma::sqrt(arma::vec(arma::trans(arma::sum(arma::square(marker_mat), 0))));

    arma::sp_mat raw_stats = arma::trans(arma::sp_mat(X * T));
    arma::mat stats;
    if (alpha == 0) {
        stats = raw_stats;
    } else {
        stats = actionet::compute_network_diffusion_fast(G, raw_stats, thread_no, alpha, max_it) *
                arma::diagmat(arma::vec(arma::trans(arma::sum(raw_stats))));
    }

    for (int i = 0; i < stats.n_rows; i++) {
        for (int j = 0; j < stats.n_cols; j++) {
            double stat = stats(i, j);
            double z = (stat - mu(i) * w1(j)) / (sigma(i) * w2(j));
            stats(i, j) = z;
        }
    }

    return (stats);
}

arma::mat aggregate_genesets_weighted_enrichment_permutation(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                             int network_normalization_method,
                                                             int expression_normalization_method,
                                                             int gene_scaling_method,
                                                             double pre_alpha, double post_alpha, int thread_no,
                                                             int perm_no) {
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
        T = actionet::compute_network_diffusion_Chebyshev(P, T_t, thread_no, pre_alpha);
        T = arma::trans(T);
    }

    if (gene_scaling_method != 0) {
        if (gene_scaling_method > 0) {
            T = arma::normalise(T, gene_scaling_method, 1);
        } else {
            arma::mat T_t = arma::trans(T);
            T = normalize_scores(T_t, -gene_scaling_method, thread_no);
            T = arma::trans(T);
        }
    }

    arma::sp_mat X = arma::trans(marker_mat);

    arma::mat Y = T;

    arma::mat stats = spmat_mat_product(X, Y);

    arma::mat E = arma::zeros(size(stats));
    arma::mat Esq = arma::zeros(arma::size(stats));
    for (int k = 0; k < perm_no; k++) {
        arma::uvec perm = arma::randperm(Y.n_rows);
        arma::mat Y_perm = Y.rows(perm);
        arma::mat rand_stats = spmat_mat_product(X, Y_perm);

        arma::mat delta = (rand_stats - stats);
        E += delta;
        Esq += arma::square(delta);
    }
    arma::mat mu = stats + E / perm_no;
    arma::mat sigma = arma::sqrt((Esq - arma::square(E) / perm_no) / (perm_no - 1));
    arma::mat marker_stats = arma::trans((stats - mu) / sigma);

    marker_stats.replace(arma::datum::nan, 0);

    arma::mat marker_stats_smoothed = marker_stats; // zscore(marker_stats, thread_no);
    if (post_alpha != 0) {
        stdout_printf("Post-smoothing expression values ... ");
        marker_stats_smoothed = actionet::compute_network_diffusion_Chebyshev(P, marker_stats_smoothed, thread_no,
                                                                              post_alpha);
        stdout_printf("done\n");
        FLUSH;
    }

    return (marker_stats_smoothed);
}

arma::mat aggregate_genesets_weighted_enrichment(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                 int network_normalization_method, int expression_normalization_method,
                                                 int gene_scaling_method, double pre_alpha, double post_alpha,
                                                 int thread_no) {

    if (S.n_rows != marker_mat.n_rows) {
        stdout_printf("Number of genes in the expression matrix (S) and marker matrix (marker_mat) do not match\n");
        FLUSH;
        return (arma::mat());
    }
    if (S.n_cols != G.n_rows) {
        stdout_printf("Number of cell in the expression matrix (S) and cell network (G) do not match\n");
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
        T = actionet::compute_network_diffusion_Chebyshev(P, T_t, thread_no, pre_alpha);
        T = arma::trans(T);
    }

    if (gene_scaling_method != 0) {
        if (gene_scaling_method > 0) {
            T = arma::normalise(T, gene_scaling_method, 1);
        } else {
            arma::mat T_t = arma::trans(T);
            T = normalize_scores(T_t, -gene_scaling_method, thread_no);
            T = arma::trans(T);
        }
    }

    arma::mat marker_stats;
    if (gene_scaling_method >= 0) {
        arma::field<arma::mat> res = actionet::assess_enrichment(T, marker_mat, thread_no);
        marker_stats = arma::trans(res(0));
    } else {
        arma::vec w = arma::vec(arma::sqrt(arma::trans(arma::sum(arma::square(marker_mat), 0))));
        w.replace(0.0, 1.0);
        arma::mat w_mat = arma::diagmat(1.0 / w);
        arma::sp_mat marker_mat_t = arma::trans(marker_mat);
        marker_stats = arma::trans(spmat_mat_product_parallel(marker_mat_t, T, thread_no)) * w_mat;
    }
    marker_stats.replace(arma::datum::nan, 0);

    arma::mat marker_stats_smoothed = marker_stats; // zscore(marker_stats, thread_no);
    if (post_alpha != 0) {
        stdout_printf("Post-smoothing expression values ... ");
        marker_stats_smoothed = actionet::compute_network_diffusion_Chebyshev(P, marker_stats_smoothed, thread_no,
                                                                              post_alpha);
        stdout_printf("done\n");
        FLUSH;
    }

    return (marker_stats_smoothed);
}

arma::mat compute_markers_eigengene(arma::mat &S, arma::sp_mat &marker_mat, int normalization, int thread_no) {
    arma::mat St = arma::trans(S); // cell x gene

    arma::mat Z;
    if (normalization == 0) {
        Z = zscore(St, thread_no);
    } else if (normalization == 1) {
        // RIN_transform removed from package.
        Z = RIN_transform(St, thread_no);
    } else // default to z-score
    {
        Z = zscore(St, thread_no);
    }

    arma::mat stats = arma::zeros(Z.n_rows, marker_mat.n_cols);

    mini_thread::parallelFor(
            0, marker_mat.n_cols, [&](size_t i) {
                arma::vec v = arma::vec(marker_mat.col(i));
                arma::uvec idx = arma::find(v != 0);
                arma::vec w = v(idx);
                arma::mat subZ = Z.cols(idx);
                subZ.each_row() %= trans(w);
                double denom = std::sqrt(arma::sum(arma::sum(arma::cov(subZ))));
                arma::vec z = arma::sum(subZ, 1) / denom;

                arma::field<arma::mat> SVD_results = actionet::HalkoSVD(subZ, 1, 5, 0, 0);
                arma::vec u = SVD_results(0);
                if (dot(u, z) < 0) // orient
                {
                    u = -u;
                }

                u = u * stddev(z) / stddev(u);

                stats.col(i) = u;
            },
            thread_no);

    return (stats);
}