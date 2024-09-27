#include "tools/matrix_transform.hpp"
#include "utils_internal/utils_stats.hpp"

// Additional arma::sp_mat-only parameters for normalizeGraph(): fill_diag_if_empty, fill_val
arma::sp_mat normalize_matrix_internal(arma::sp_mat X, const unsigned int p, const unsigned int dim,
                                       const bool fill_diag_if_empty = false, const double fill_val = 1.0) {
    if (p > 1) {
        stderr_printf("`p` ignored for sparse input");
    }

    arma::vec sum_vec = (dim == 0) ? arma::zeros(X.n_cols) : arma::zeros(X.n_rows);

    if (dim == 0) {
        for (arma::sp_mat::const_iterator it = X.begin(); it != X.end(); ++it) {
            sum_vec[it.col()] += std::abs(*it);
        }
    }
    else {
        for (arma::sp_mat::const_iterator it = X.begin(); it != X.end(); ++it) {
            sum_vec[it.row()] += std::abs(*it);
        }
    }

    arma::uvec zidx = arma::find(sum_vec == 0);
    sum_vec.transform([](double val) { return (val == 0 ? 1 : val); });

    if (dim == 0) {
        for (arma::sp_mat::iterator it = X.begin(); it != X.end(); ++it) {
            const double w = sum_vec[it.col()];
            (*it) /= w;
        }
    }
    else {
        for (arma::sp_mat::iterator it = X.begin(); it != X.end(); ++it) {
            const double w = sum_vec[it.row()];
            (*it) /= w;
        }
    }

    if (fill_diag_if_empty) {
        for (size_t i = 0; i < zidx.n_elem; i++) {
            const size_t k = zidx(i);
            X(k, k) = fill_val;
        }
    }

    return (X);
}

arma::mat normalize_matrix_internal(arma::mat X, unsigned int p, unsigned int dim) {
    if (dim != 0 && dim != 1) {
        throw std::invalid_argument("Invalid dimension");
    }

    X = arma::normalise(X, p, dim);

    return (X);
}

arma::mat scale_matrix_internal(arma::mat S, const arma::vec& v, unsigned int dim) {
    if (dim == 1) {
        S.each_col() %= v;
    }
    else {
        S.each_row() %= v.t();
    }

    return (S);
}


arma::sp_mat scale_matrix_internal(arma::sp_mat S, const arma::vec& v, unsigned int dim) {
    arma::sp_mat::iterator it = S.begin();
    const arma::sp_mat::iterator it_end = S.end();

    if (dim == 1) {
        for (; it != it_end; ++it) {
            (*it) *= v[it.row()];
        }
    }
    else {
        for (; it != it_end; ++it) {
            (*it) *= v[it.col()];
        }
    }

    return (S);
}

namespace actionet {
    template <typename T>
    T normalizeMatrix(T& X, unsigned int p, unsigned int dim) {
        if (dim != 0 && dim != 1) {
            throw std::invalid_argument("Invalid dimension");
        }
        if (p > 0) {
            X = normalize_matrix_internal(X, p, dim);
        }

        return (X);
    }

    template arma::mat normalizeMatrix<arma::mat>(arma::mat& X, unsigned int p, unsigned int dim);
    template arma::sp_mat normalizeMatrix<arma::sp_mat>(arma::sp_mat& X, unsigned int p, unsigned int dim);

    template <typename T>
    T scaleMatrix(T& X, arma::vec& v, unsigned int dim) {
        size_t mat_dim = (dim == 0) ? X.n_cols : X.n_rows;

        if (mat_dim != v.n_elem) {
            throw std::invalid_argument("Sizes of 'S' and 'v' do not match for given 'dim'");
        }

        X = scale_matrix_internal(X, v, dim);

        return (X);
    }

    template arma::mat scaleMatrix<arma::mat>(arma::mat& X, arma::vec& v, unsigned int dim);
    template arma::sp_mat scaleMatrix<arma::sp_mat>(arma::sp_mat& X, arma::vec& v, unsigned int dim);

    // Graph pre-normalization for PageRank
    // norm_type 0/1 is standard unit normalization columns/rows with edge-cases.
    // norm_type 2 symmetrizes graph, i.e. colsums == rowsums.
    arma::sp_mat normalizeGraph(arma::sp_mat& G, int norm_type) {
        arma::sp_mat P = G;
        switch (norm_type) {
            case 0: // Column-normalize
            case 1: // Row-normalize
            {
                P = normalize_matrix_internal(P, 1, norm_type, true, 1.0);
            }
            break;
            case 2: // Symmetrize
            {
                arma::vec row_sums = arma::zeros(G.n_rows);
                arma::vec col_sums = arma::zeros(G.n_cols);

                arma::sp_mat::iterator it = G.begin();
                arma::sp_mat::iterator it_end = G.end();

                for (; it != it_end; ++it) {
                    col_sums[it.col()] += (*it);
                    row_sums[it.row()] += (*it);
                }

                row_sums.transform([](double val) { return (val == 0 ? 1 : val); });
                col_sums.transform([](double val) { return (val == 0 ? 1 : val); });

                for (it = P.begin(); it != P.end(); ++it) {
                    double w = std::sqrt(row_sums[it.row()] * col_sums[it.col()]);
                    (*it) /= w;
                }
            }
            break;
            default:
                throw std::invalid_argument("Invalid 'norm_type'");
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
} // namespace actionet
