#include "tools/autocorrelation.hpp"
#include "tools/normalization.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"

namespace actionet {
    arma::field<arma::vec>
    autocorrelation_Moran_parametric(arma::mat G, arma::mat scores, int normalization_method, int thread_no) {
        // G.each_row() -= sum(G, 1); // Make row-sum == 1

        double nV = G.n_rows;
        int scores_no = scores.n_cols;
        stdout_printf("Normalizing scores (method=%d) ... ", normalization_method);
        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);
        stdout_printf("done\n");
        FLUSH;

        stdout_printf("Computing auto-correlation over network ... ");

        double W = arma::sum(arma::sum(G));
        double Wsq = W * W;

        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = nV / (W * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        arma::vec stat = arma::zeros(scores_no);
        mini_thread::parallelFor(
            0, scores_no,
            [&](unsigned int i) {
                arma::vec x = normalized_scores.col(i);
                double y = dot(x, G * x);
                stat(i) = y;
            },
            thread_no);

        stat = stat % norm_factors;

        stdout_printf("done\n");
        FLUSH;

        arma::vec mu = -arma::ones(scores_no) / (nV - 1);

        arma::mat Gsym = (G + arma::trans(G));
        double S1 = 0.5 * arma::sum(arma::sum(arma::square(Gsym)));

        arma::vec rs = arma::vec(arma::sum(G, 1));
        arma::vec cs = arma::vec(arma::trans(arma::sum(G, 0)));
        arma::vec sg = rs + cs;
        double S2 = arma::sum(arma::square(sg));

        arma::mat normalized_scores_sq = arma::square(normalized_scores);
        arma::vec S3_vec = arma::trans((arma::sum(arma::square(normalized_scores_sq), 0) / nV) /
                                       (arma::square(arma::sum(normalized_scores_sq, 0) / nV)));
        double S4 = (nV * (nV - 3) + 3) * S1 - nV * S2 + 3 * Wsq;
        double S5 = (nV * (nV - 1)) * S1 - 2 * nV * S2 + 6 * Wsq;

        double k1 = (nV * S4) / ((nV - 1) * (nV - 2) * (nV - 3) * Wsq);
        double k2 = S5 / ((nV - 1) * (nV - 2) * (nV - 3) * Wsq);

        arma::vec sigma_sq = k1 - k2 * S3_vec - arma::square(mu);
        arma::vec sigma = arma::sqrt(sigma_sq);

        arma::vec zscores = (stat - mu) / sigma;

        // Summary stats
        arma::field<arma::vec> results(4);
        results(0) = stat;
        results(1) = zscores;
        results(2) = mu;
        results(3) = sigma;

        return (results);
    }

    arma::field<arma::vec>
    autocorrelation_Moran_parametric(arma::sp_mat G, arma::mat scores, int normalization_method, int thread_no) {

        double nV = G.n_rows;
        int scores_no = scores.n_cols;
        stdout_printf("Normalizing scores (method=%d) ... ", normalization_method);
        arma::mat normalized_scores =
                normalize_scores(scores, normalization_method, thread_no);
        stdout_printf("done\n");
        FLUSH;

        stdout_printf("Computing auto-correlation over network ... ");

        double W = arma::sum(arma::sum(G));
        double Wsq = W * W;

        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = nV / (W * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        arma::vec stat = arma::zeros(scores_no);
        mini_thread::parallelFor(
            0, scores_no,
            [&](unsigned int i) {
                arma::vec x = normalized_scores.col(i);
                double y = dot(x, spmat_vec_product(G, x));
                stat(i) = y;
            },
            thread_no);

        stat = stat % norm_factors;

        stdout_printf("done\n");
        FLUSH;

        arma::vec mu = -arma::ones(scores_no) / (nV - 1);

        arma::sp_mat Gsym = (G + arma::trans(G));
        double S1 = 0.5 * arma::sum(arma::sum(arma::square(Gsym)));

        arma::vec rs = arma::vec(arma::sum(G, 1));
        arma::vec cs = arma::vec(arma::trans(arma::sum(G, 0)));
        arma::vec sg = rs + cs;
        double S2 = arma::sum(arma::square(sg));

        arma::mat normalized_scores_sq = arma::square(normalized_scores);
        arma::vec S3_vec = arma::trans((arma::sum(arma::square(normalized_scores_sq), 0) / nV) /
                                       (arma::square(arma::sum(normalized_scores_sq, 0) / nV)));
        double S4 = (nV * (nV - 3) + 3) * S1 - nV * S2 + 3 * Wsq;
        double S5 = (nV * (nV - 1)) * S1 - 2 * nV * S2 + 6 * Wsq;

        double k1 = (nV * S4) / ((nV - 1) * (nV - 2) * (nV - 3) * Wsq);
        double k2 = S5 / ((nV - 1) * (nV - 2) * (nV - 3) * Wsq);

        arma::vec sigma_sq = k1 - k2 * S3_vec - arma::square(mu);
        arma::vec sigma = arma::sqrt(sigma_sq);

        arma::vec zscores = (stat - mu) / sigma;

        // Summary stats
        arma::field<arma::vec> results(4);
        results(0) = stat;
        results(1) = zscores;
        results(2) = mu;
        results(3) = sigma;

        return (results);
    }

    arma::field<arma::vec>
    autocorrelation_Moran(arma::mat G, arma::mat scores, int normalization_method, int perm_no, int thread_no) {
        int nV = G.n_rows;
        int scores_no = scores.n_cols;
        stdout_printf("Normalizing scores (method=%d) ... ", normalization_method);
        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);
        stdout_printf("done\n");
        FLUSH;

        stdout_printf("Computing auto-correlation over network ... ");

        double W = arma::sum(arma::sum(G));
        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = nV / (W * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        arma::vec stat = arma::zeros(scores_no);
        mini_thread::parallelFor(
            0, scores_no,
            [&](unsigned int i) {
                arma::vec x = normalized_scores.col(i);
                double y = dot(x, G * x);
                stat(i) = y;
            },
            thread_no);

        arma::vec mu = arma::zeros(scores_no);
        arma::vec sigma = arma::zeros(scores_no);
        arma::vec z = arma::zeros(scores_no);
        if (0 < perm_no) {
            arma::mat rand_stats = arma::zeros(scores_no, perm_no);
            mini_thread::parallelFor(
                0, perm_no,
                [&](unsigned int j) {
                    arma::uvec perm = arma::randperm(nV);
                    arma::mat score_permuted = normalized_scores.rows(perm);

                    arma::vec v = arma::zeros(scores_no);
                    for (int i = 0; i < scores_no; i++) {
                        arma::vec rand_x = score_permuted.col(i);
                        v(i) = arma::dot(rand_x, G * rand_x);
                    }
                    rand_stats.col(j) = v;
                },
                thread_no);

            mu = arma::mean(rand_stats, 1);
            sigma = arma::stddev(rand_stats, 0, 1);
            z = (stat - mu) / sigma;
            z.replace(arma::datum::nan, 0);
        }
        stdout_printf("done\n");
        FLUSH;

        // Summary stats
        arma::field<arma::vec> results(4);
        results(0) = stat % norm_factors;
        results(1) = z;
        results(2) = mu;
        results(3) = sigma;
        return (results);
    }

    arma::field<arma::vec>
    autocorrelation_Moran(arma::sp_mat G, arma::mat scores, int normalization_method, int perm_no, int thread_no) {
        int nV = G.n_rows;
        int scores_no = scores.n_cols;

        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);

        stdout_printf("Computing auto-correlation over network ... ");
        double W = arma::sum(arma::sum(G));
        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = nV / (W * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        arma::vec stat = arma::zeros(scores_no);
        mini_thread::parallelFor(
            0, scores_no,
            [&](unsigned int i) {
                arma::vec x = normalized_scores.col(i);
                double y = arma::dot(x, spmat_vec_product(G, x));
                stat(i) = y;
            },
            thread_no);

        arma::vec mu = arma::zeros(scores_no);
        arma::vec sigma = arma::zeros(scores_no);
        arma::vec z = arma::zeros(scores_no);
        if (0 < perm_no) {
            arma::mat rand_stats = arma::zeros(scores_no, perm_no);
            mini_thread::parallelFor(
                0, perm_no,
                [&](unsigned int j) {
                    arma::uvec perm = arma::randperm(nV);
                    arma::mat score_permuted = normalized_scores.rows(perm);

                    arma::vec v = arma::zeros(scores_no);
                    for (int i = 0; i < scores_no; i++) {
                        arma::vec rand_x = score_permuted.col(i);
                        v(i) = arma::dot(rand_x, spmat_vec_product(G, rand_x));
                    }
                    rand_stats.col(j) = v;
                },
                thread_no);

            mu = arma::mean(rand_stats, 1);
            sigma = arma::stddev(rand_stats, 0, 1);
            z = (stat - mu) / sigma;
            z.replace(arma::datum::nan, 0);
        }
        stdout_printf("done\n");
        FLUSH;

        // Summary stats
        arma::field<arma::vec> results(4);
        results(0) = stat % norm_factors;
        results(1) = z;
        results(2) = mu;
        results(3) = sigma;

        return (results);
    }

    arma::field<arma::vec>
    autocorrelation_Geary(arma::mat G, arma::mat scores, int normalization_method, int perm_no, int thread_no) {
        int nV = G.n_rows;
        int scores_no = scores.n_cols;
        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);

        stdout_printf("Computing auto-correlation over network ... ");
        double W = arma::sum(arma::sum(G));
        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = (nV - 1) / ((2 * W) * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        // Compute graph Laplacian
        arma::vec d = arma::vec(arma::trans(arma::sum(G)));
        arma::mat L(-G);
        L.diag() = d;

        arma::vec stat = arma::zeros(scores_no);
        mini_thread::parallelFor(
            0, scores_no,
            [&](unsigned int i) {
                arma::vec x = normalized_scores.col(i);
                double y = arma::dot(x, L * x);
                stat(i) = y;
            },
            thread_no);

        arma::vec mu = arma::zeros(scores_no);
        arma::vec sigma = arma::zeros(scores_no);
        arma::vec z = arma::zeros(scores_no);
        if (0 < perm_no) {
            arma::mat rand_stats = arma::zeros(scores_no, perm_no);
            mini_thread::parallelFor(
                0, perm_no,
                [&](unsigned int j) {
                    arma::uvec perm = arma::randperm(nV);
                    arma::mat score_permuted = normalized_scores.rows(perm);

                    arma::vec v = arma::zeros(scores_no);
                    for (int i = 0; i < scores_no; i++) {
                        arma::vec rand_x = score_permuted.col(i);
                        v(i) = arma::dot(rand_x, L * rand_x);
                    }
                    rand_stats.col(j) = v;
                },
                thread_no);

            mu = arma::mean(rand_stats, 1);
            sigma = arma::stddev(rand_stats, 0, 1);
            z = (stat - mu) / sigma;
            z.replace(arma::datum::nan, 0);
        }
        stdout_printf("done\n");
        FLUSH;

        // Summary stats
        arma::field<arma::vec> results(4);
        results(0) = stat % norm_factors;
        results(1) = -z;
        results(2) = mu;
        results(3) = sigma;
        return (results);
    }

    arma::field<arma::vec>
    autocorrelation_Geary(arma::sp_mat G, arma::mat scores, int normalization_method, int perm_no, int thread_no) {
        int nV = G.n_rows;
        int scores_no = scores.n_cols;
        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);

        stdout_printf("Computing auto-correlation over network ... ");
        double W = arma::sum(arma::sum(G));
        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = (nV - 1) / ((2 * W) * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        // Compute graph Laplacian
        arma::vec d = arma::vec(arma::trans(arma::sum(G)));
        arma::sp_mat L(-G);
        L.diag() = d;

        arma::vec stat = arma::zeros(scores_no);
        mini_thread::parallelFor(
            0, scores_no,
            [&](unsigned int i) {
                arma::vec x = normalized_scores.col(i);
                double y = arma::dot(x, spmat_vec_product(L, x));
                stat(i) = y;
            },
            thread_no);

        arma::vec mu = arma::zeros(scores_no);
        arma::vec sigma = arma::zeros(scores_no);
        arma::vec z = arma::zeros(scores_no);
        if (0 < perm_no) {
            arma::mat rand_stats = arma::zeros(scores_no, perm_no);
            mini_thread::parallelFor(
                0, perm_no,
                [&](unsigned int j) {
                    arma::uvec perm = arma::randperm(nV);
                    arma::mat score_permuted = normalized_scores.rows(perm);

                    arma::vec v = arma::zeros(scores_no);
                    for (int i = 0; i < scores_no; i++) {
                        arma::vec rand_x = score_permuted.col(i);
                        v(i) = arma::dot(rand_x, spmat_vec_product(L, rand_x));
                    }
                    rand_stats.col(j) = v;
                },
                thread_no);

            mu = arma::mean(rand_stats, 1);
            sigma = arma::stddev(rand_stats, 0, 1);
            z = (stat - mu) / sigma;
            z.replace(arma::datum::nan, 0);
        }
        stdout_printf("done\n");
        FLUSH;

        // Summary stats
        arma::field<arma::vec> results(4);
        results(0) = stat % norm_factors;
        results(1) = -z;
        results(2) = mu;
        results(3) = sigma;

        return (results);
    }
} // namespace actionet
