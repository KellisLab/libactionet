#include "utils_graph.hpp"

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
