#include "annotation/specificity.hpp"
#include "utils_internal/utils_matrix.hpp"

arma::field<arma::mat> getProbsObs(const arma::mat& S, arma::mat& Ht, int thread_no) {
    arma::mat Sb = S;
    arma::uvec nnz_idx = arma::find(Sb > 0);
    (Sb(nnz_idx)).ones();

    arma::vec col_p = arma::vec(arma::trans(arma::sum(Sb, 0)));
    arma::vec row_p = arma::vec(arma::sum(Sb, 1));
    arma::vec row_factor = arma::vec(arma::sum(S, 1));

    arma::field<arma::mat> out(4);
    out(0) = row_factor / row_p; // mean of nonzero elements
    out(1) = row_p / S.n_cols;
    out(2) = col_p / S.n_rows;
    out(3) = S * Ht;

    return (out);
}

arma::field<arma::mat> getProbsObs(const arma::sp_mat& S, arma::mat& Ht, int thread_no) {
    // Heuristic optimization! Shall add parallel for later on
    arma::vec row_p = arma::zeros(S.n_rows);
    arma::vec col_p = arma::zeros(S.n_cols);
    arma::vec row_factor = arma::zeros(S.n_rows);

    arma::sp_mat::const_iterator it = S.begin();
    arma::sp_mat::const_iterator it_end = S.end();
    for (; it != it_end; ++it) {
        col_p[it.col()]++;
        row_p[it.row()]++;
        row_factor[it.row()] += (*it);
    }

    arma::field<arma::mat> out(4);
    out(0) = row_factor / row_p; // mean of nonzero elements
    out(1) = row_p / S.n_cols;
    out(2) = col_p / S.n_rows;
    out(3) = spmat_mat_product_parallel(S, Ht, thread_no);

    return (out);
}

namespace actionet {
    template <typename T>
    arma::field<arma::mat> compute_feature_specificity(T& S, arma::mat& H, int thread_no) {
        stdout_printf("Computing feature specificity ... ");

        // make sure all values are positive
        double min_val = S.min();
        S.for_each([min_val](arma::mat::elem_type& val) { val -= min_val; });

        arma::mat Ht = arma::trans(H);
        Ht.each_col([](arma::vec& h) {
            double mu = arma::mean(h);
            h /= (mu == 0) ? 1 : mu;
        }); // For numerical stability

        arma::field<arma::mat> p = getProbsObs(S, Ht, thread_no);

        arma::vec row_factor = p(0);
        arma::vec row_p = p(1);
        arma::vec col_p = p(2);
        arma::mat Obs = p(3);

        double rho = arma::mean(col_p);
        arma::vec beta = col_p / rho; // Relative density compared to the overall density
        arma::mat Gamma = Ht;
        arma::vec a(H.n_rows);
        for (int i = 0; i < H.n_rows; i++) {
            Gamma.col(i) %= beta;
            a(i) = arma::max(Gamma.col(i));
        }

        arma::mat Exp = (row_p % row_factor) * arma::sum(Gamma, 0);
        arma::mat Nu = (row_p % arma::square(row_factor)) * arma::sum(arma::square(Gamma), 0);
        arma::mat A = (row_factor * arma::trans(a));
        arma::mat Lambda = Obs - Exp;

        arma::mat logPvals_lower = arma::square(Lambda) / (2 * Nu);
        arma::uvec uidx = arma::find(Lambda >= 0);
        logPvals_lower(uidx) = arma::zeros(uidx.n_elem);
        logPvals_lower.replace(arma::datum::nan, 0); // replace each NaN with 0

        arma::mat logPvals_upper = arma::square(Lambda) / (2 * (Nu + (Lambda % A / 3)));
        arma::uvec lidx = arma::find(Lambda <= 0);
        logPvals_upper(lidx) = arma::zeros(lidx.n_elem);
        logPvals_upper.replace(arma::datum::nan, 0); // replace each NaN with 0

        logPvals_lower /= log(10);
        logPvals_upper /= log(10);

        stdout_printf("done\n");
        FLUSH;

        arma::field<arma::mat> res(3);
        res(0) = Obs / Ht.n_rows;
        res(1) = logPvals_upper;
        res(2) = logPvals_lower;

        return (res);
    }

    template arma::field<arma::mat> compute_feature_specificity<arma::mat>(arma::mat& S, arma::mat& H, int thread_no);
    template arma::field<arma::mat> compute_feature_specificity<arma::sp_mat>(
        arma::sp_mat& S, arma::mat& H, int thread_no);

    template <typename T>
    arma::field<arma::mat> compute_feature_specificity(T& S, arma::uvec sample_assignments, int thread_no) {
        arma::mat H(arma::max(sample_assignments), S.n_cols);

        for (int i = 1; i <= arma::max(sample_assignments); i++) {
            arma::vec v = arma::zeros(S.n_cols);
            arma::uvec idx = arma::find(sample_assignments == i);
            v(idx) = arma::ones(idx.n_elem);
            H.row(i - 1) = arma::trans(v);
        }

        arma::field<arma::mat> res = compute_feature_specificity(S, H, thread_no);

        return (res);
    }

    template arma::field<arma::mat> compute_feature_specificity<arma::mat>(
        arma::mat& S, arma::uvec sample_assignments, int thread_no);
    template arma::field<arma::mat> compute_feature_specificity<arma::sp_mat>(
        arma::sp_mat& S, arma::uvec sample_assignments, int thread_no);
} // namespace actionet
