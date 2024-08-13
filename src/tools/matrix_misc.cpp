#include "tools/matrix_misc.hpp"

namespace ACTIONet {

    arma::mat compute_grouped_rowsums(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments) {

        arma::uvec lv_vec = arma::conv_to<arma::uvec>::from(arma::unique(sample_assignments));
        arma::mat pb = arma::zeros(S.n_rows, lv_vec.n_elem);

        arma::sp_mat::const_iterator it = S.begin();
        arma::sp_mat::const_iterator it_end = S.end();
        for (; it != it_end; ++it) {
            int i = it.row();
            int j = sample_assignments[it.col()] - 1;
            pb(i, j) += (*it);
        }

        return (pb);
    }

    arma::mat compute_grouped_rowsums(arma::mat &S, arma::Col<unsigned long long> sample_assignments) {

        arma::uvec lv_vec = arma::conv_to<arma::uvec>::from(arma::unique(sample_assignments));
        arma::mat pb = arma::zeros(S.n_rows, lv_vec.n_elem);

        for (int j = 0; j < pb.n_cols; j++) {
            arma::uvec idx = arma::find(sample_assignments == (j + 1));
            if (idx.n_elem == 0) continue;

            if (idx.n_elem > 1) {
                arma::mat subS = S.cols(idx);
                pb.col(j) = arma::sum(subS, 1);
            } else {
                pb.col(j) = S.col(idx(0));
            }
        }

        return (pb);
    }

    arma::mat compute_grouped_rowmeans(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments) {

        arma::mat pb = compute_grouped_rowsums(S, sample_assignments);

        for (int j = 0; j < pb.n_cols; j++) {
            arma::uvec idx = arma::find(sample_assignments == (j + 1));
            pb.col(j) /= std::max(1, (int) idx.n_elem);
        }

        return (pb);
    }

    arma::mat compute_grouped_rowmeans(arma::mat &S, arma::Col<unsigned long long> sample_assignments) {

        arma::mat pb = compute_grouped_rowsums(S, sample_assignments);

        for (int j = 0; j < pb.n_cols; j++) {
            arma::uvec idx = arma::find(sample_assignments == (j + 1));
            pb.col(j) /= std::max(1, (int) idx.n_elem);
        }

        return (pb);
    }

    arma::mat compute_grouped_rowvars(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments) {

        arma::mat pb_mu = compute_grouped_rowmeans(S, sample_assignments);
        arma::mat pb = arma::zeros(pb_mu.n_rows, pb_mu.n_cols);
        arma::mat pbz = arma::zeros(pb_mu.n_rows, pb_mu.n_cols);

        arma::sp_mat::const_iterator it = S.begin();
        arma::sp_mat::const_iterator it_end = S.end();
        for (; it != it_end; ++it) {
            int i = it.row();
            int j = sample_assignments[it.col()] - 1;
            double num = (*it) - pb_mu(i, j);
            pb(i, j) += num * num;
            pbz(i, j) += 1;
        }

        for (int j = 0; j < pb.n_cols; j++) {
            arma::uvec idx = arma::find(sample_assignments == (j + 1));
            int nnz = (int) idx.n_elem;
            for (int i = 0; i < pb.n_rows; i++) {
                int nz = (int) idx.n_elem - pbz(i, j);
                pb(i, j) += nz * pb_mu(i, j) * pb_mu(i, j);
            }
            pb.col(j) /= std::max(1, nnz - 1);
        }

        return (pb);
    }

    arma::mat compute_grouped_rowvars(arma::mat &S, arma::Col<unsigned long long> sample_assignments) {

        arma::uvec lv_vec = arma::conv_to<arma::uvec>::from(unique(sample_assignments));
        arma::mat pb = arma::zeros(S.n_rows, lv_vec.n_elem);

        for (int j = 0; j < pb.n_cols; j++) {
            arma::uvec idx = arma::find(sample_assignments == (j + 1));

            if (idx.n_elem == 0) continue;

            if (idx.n_elem > 1) {
                arma::mat subS = S.cols(idx);
                pb.col(j) = arma::var(subS, 0, 1);
            } else {
                pb.col(j) = arma::zeros(pb.n_rows);
            }
        }
        return (pb);
    }

}  // namespace ACTIONet
