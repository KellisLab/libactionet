#include "specificity_ext.h"

arma::field<arma::mat> compute_feature_specificity_bin(arma::sp_mat &Sb, arma::mat &H, int thread_no) {
    stdout_printf("Computing feature specificity ... ");
    arma::field<arma::mat> res(3);

    arma::mat Ht = arma::trans(H);
    Ht.each_col([](arma::vec &h) {
        double mu = arma::mean(h);
        h /= (mu == 0) ? 1 : mu;
    }); // For numerical stability

    // Heuristic optimization! Shall add parallel for later on
    arma::vec row_p = arma::zeros(Sb.n_rows);
    arma::vec col_p = arma::zeros(Sb.n_cols);
    arma::sp_mat::const_iterator it = Sb.begin();
    arma::sp_mat::const_iterator it_end = Sb.end();
    for (; it != it_end; ++it) {
        row_p[it.row()]++;
        col_p[it.col()]++;
    }
    row_p /= Sb.n_cols;
    col_p /= Sb.n_rows;

    arma::mat Obs = spmat_mat_product_parallel(Sb, Ht, thread_no);

    double rho = arma::mean(col_p);
    arma::vec beta = col_p / rho; // Relative density compared to the overall density
    arma::mat Gamma = Ht;
    arma::vec a(H.n_rows);
    for (int i = 0; i < H.n_rows; i++) {
        Gamma.col(i) %= beta;
        a(i) = arma::max(Gamma.col(i));
    }

    arma::mat Exp = row_p * arma::sum(Gamma, 0);
    arma::mat Nu = row_p * arma::sum(arma::square(Gamma), 0);
    arma::mat Lambda = Obs - Exp;

    arma::mat logPvals_lower = arma::square(Lambda) / (2 * Nu);
    arma::uvec uidx = arma::find(Lambda >= 0);
    logPvals_lower(uidx) = arma::zeros(uidx.n_elem);
    logPvals_lower.replace(arma::datum::nan, 0); // replace each NaN with 0

    arma::mat Lambda_scaled = Lambda;
    for (int j = 0; j < Lambda_scaled.n_cols; j++) {
        Lambda_scaled.col(j) *= (a(j) / 3);
    }
    arma::mat logPvals_upper = arma::square(Lambda) / (2 * (Nu + Lambda_scaled));
    arma::uvec lidx = arma::find(Lambda <= 0);
    logPvals_upper(lidx) = arma::zeros(lidx.n_elem);
    logPvals_upper.replace(arma::datum::nan, 0); // replace each NaN with 0

    logPvals_lower /= std::log(10);
    logPvals_upper /= std::log(10);
    stdout_printf("done\n");
    FLUSH;

    res(0) = Obs / Ht.n_rows;
    res(1) = logPvals_upper;
    res(2) = logPvals_lower;

    return (res);
}