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