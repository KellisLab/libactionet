#include "tools/normalization.hpp"

namespace ACTIONet {

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
    
} // namespace ACTIONet

