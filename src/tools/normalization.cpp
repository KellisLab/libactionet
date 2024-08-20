#include "tools/normalization.hpp"
#include "utils_internal/utils_stats.hpp"

namespace actionet {

    // TODO: Why copy X?
    arma::mat normalize_mat(arma::mat &X, int normalization, int dim) {
        arma::mat X_norm = X;
        if (normalization == 1) {
            X_norm = arma::normalise(X_norm, 1, dim);
        }
        if (normalization == 2) {
            X_norm = arma::normalise(X_norm, 2, dim);
        }
        if (normalization == -1) {
            X_norm = zscore(X_norm, dim);
        }

        return (X_norm);
    }

    // TODO: Why copy X?
    arma::sp_mat normalize_mat(arma::sp_mat &X, int normalization, int dim) {
        arma::sp_mat X_norm = X;
        if (normalization == 1) {
            X_norm = arma::normalise(X_norm, 1, dim);
        }
        if (normalization == 2) {
            X_norm = arma::normalise(X_norm, 2, dim);
        }
        return (X_norm);
    }

    arma::sp_mat normalize_adj(arma::sp_mat &G, int norm_type) {
        arma::vec row_sums = arma::zeros(G.n_rows);
        arma::vec col_sums = arma::zeros(G.n_cols);

        arma::sp_mat::iterator it = G.begin();
        arma::sp_mat::iterator it_end = G.end();
        for (; it != it_end; ++it) {
            col_sums[it.col()] += (*it);
            row_sums[it.row()] += (*it);
        }
        arma::uvec idxr = arma::find(row_sums == 0);
        arma::uvec idxc = arma::find(col_sums == 0);

        row_sums.transform([](double val) { return (val == 0 ? 1 : val); });
        col_sums.transform([](double val) { return (val == 0 ? 1 : val); });

        // Update
        arma::sp_mat P = G;
        if (norm_type == 0) // Column-normalize
        {
            for (it = P.begin(); it != P.end(); ++it) {
                double w = col_sums[it.col()];
                (*it) /= w;
            }
            for (int k = 0; k < idxc.n_elem; k++) {
                int j = idxc(k);
                P(j, j) = 1.0;
            }
        } else if (norm_type == 1) // Row-normalize
        {
            for (it = P.begin(); it != P.end(); ++it) {
                double w = row_sums[it.row()];
                (*it) /= w;
            }
            for (int k = 0; k < idxr.n_elem; k++) {
                int i = idxr(k);
                P(i, i) = 1.0;
            }
        } else if (norm_type == 2) {
            for (it = P.begin(); it != P.end(); ++it) {
                double w = std::sqrt(row_sums[it.row()] * col_sums[it.col()]);
                (*it) /= w;
            }
        }

        return (P);
    }

    // TODO: Rename and consolidate with above functions
    arma::mat normalize_scores(arma::mat scores, int method, int thread_no) {
        arma::mat normalized_scores(size(scores));
        switch (method) {
            case 0: //"none"
            {
                normalized_scores = scores;
                break;
            }
            case 1: //"zscore"
            {
                normalized_scores = zscore(scores, 0, thread_no);
                break;
            }
            case 2: //"RINT" (nonparametric)
            {
                stderr_printf("RINT remove. Returning mat.\n");
                FLUSH;
                normalized_scores = scores;
                break;
            }
            case 3: //"robust_zscore" (kinda hack!)
            {
                normalized_scores = robust_zscore(scores, 0, thread_no);
                break;
            }
            case 4: // mean centering
            {
                normalized_scores = mean_center(scores);
                break;
            }
            default:
                stderr_printf("Unknown normalization method\n");
                FLUSH;
                normalized_scores = scores;
        }
        return (normalized_scores);
    }

    // TODO: TF-IDF normalization (change name)
    // Formula might be wrong.
    arma::sp_mat LSI(arma::sp_mat &S, double size_factor) {
        arma::sp_mat X = S;

        arma::vec col_sum_vec = arma::zeros(X.n_cols);
        arma::vec row_sum_vec = arma::zeros(X.n_rows);

        arma::sp_mat::iterator it = X.begin();
        arma::sp_mat::iterator it_end = X.end();
        for (; it != it_end; ++it) {
            col_sum_vec(it.col()) += (*it);
            row_sum_vec(it.row()) += (*it);
        }

        arma::vec kappa = size_factor / col_sum_vec;
        arma::vec IDF = arma::log(1 + (X.n_cols / row_sum_vec));

        for (it = X.begin(); it != X.end(); ++it) {
            double x = (*it) * kappa(it.col());
            x = std::log(1 + x) * IDF(it.row());
            *it = x;
        }

        return (X);
    }

} // namespace actionet

