#include "normalization_ext.h"

arma::sp_mat renormalize_input_matrix(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments) {
    stdout_printf("Computing and normalizing pseudobulk profiles ... ");
    arma::mat pb = compute_grouped_rowmeans(S, sample_assignments);
    arma::rowvec mean_pb_t = arma::trans(arma::mean(pb, 1));

    // Aligns pseudo-bulk profiles with each other
    for (int j = 0; j < pb.n_cols; j++) {
        arma::vec x = pb.col(j);

        double num = arma::sum(mean_pb_t * x);
        double denom = arma::sum(arma::square(x));

        pb.col(j) *= (num / denom);
    }
    stdout_printf("done\n");
    FLUSH;

    // Align individual columns now
    stdout_printf("Normalizing columns ... ");

    arma::sp_mat S_norm = S;
    arma::mat pb_t = arma::trans(pb);
    for (int j = 0; j < S.n_cols; j++) {
        arma::vec x = arma::vec(S.col(j));
        arma::rowvec y_t = pb_t.row(sample_assignments(j) - 1);

        double num = arma::sum(y_t * x);
        double denom = arma::sum(arma::square(x));

        S_norm.col(j) *= (num / denom);
    }
    stdout_printf("done\n");
    FLUSH;

    return (S_norm);
}

arma::mat renormalize_input_matrix(arma::mat &S, arma::Col<unsigned long long> sample_assignments) {
    stdout_printf("Computing and normalizing pseudobulk profiles ... ");
    arma::mat pb = compute_grouped_rowmeans(S, sample_assignments);
    arma::rowvec mean_pb_t = arma::trans(arma::mean(pb, 1));

    // Aligns pseudo-bulk profiles with each other
    for (int j = 0; j < pb.n_cols; j++) {
        arma::vec x = pb.col(j);

        double num = arma::sum(mean_pb_t * x);
        double denom = arma::sum(arma::square(x));

        pb.col(j) *= (num / denom);
    }
    stdout_printf("done\n");
    FLUSH;

    // Align individual columns now
    stdout_printf("Normalizing columns ... ");

    arma::mat S_norm = S;
    arma::mat pb_t = arma::trans(pb);
    for (int j = 0; j < S.n_cols; j++) {
        arma::vec x = arma::vec(S.col(j));
        arma::rowvec y_t = pb_t.row(sample_assignments(j) - 1);

        double num = arma::sum(y_t * x);
        double denom = arma::sum(arma::square(x));

        S_norm.col(j) *= (num / denom);
    }
    stdout_printf("done\n");
    FLUSH;

    return (S_norm);
}

// TF-IDF normalization (change name)
// Formula might be wrong.
arma::sp_mat LSI(const arma::sp_mat& S, double size_factor)
{
    arma::sp_mat X = S;

    arma::vec col_sum_vec = arma::zeros(X.n_cols);
    arma::vec row_sum_vec = arma::zeros(X.n_rows);

    arma::sp_mat::iterator it = X.begin();
    arma::sp_mat::iterator it_end = X.end();
    for (; it != it_end; ++it)
    {
        col_sum_vec(it.col()) += (*it);
        row_sum_vec(it.row()) += (*it);
    }

    arma::vec kappa = size_factor / col_sum_vec;
    arma::vec IDF = arma::log(1 + (X.n_cols / row_sum_vec));

    for (it = X.begin(); it != X.end(); ++it)
    {
        double x = (*it) * kappa(it.col());
        x = std::log(1 + x) * IDF(it.row());
        *it = x;
    }

    return (X);
}