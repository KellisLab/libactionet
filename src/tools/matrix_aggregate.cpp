#include "tools/matrix_aggregate.hpp"

namespace actionet {

    arma::mat computeGroupedSums(arma::sp_mat& S, arma::vec& sample_assignments, int axis) {
        if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
            throw std::invalid_argument("Length of 'sample_assignments' must match the number of columns (axis=0) or rows (axis=1) of S.");
        }
        arma::vec lv_vec = arma::unique(sample_assignments);
        if (axis == 0) { // rowwise (default)
            arma::mat pb = arma::zeros(S.n_rows, lv_vec.n_elem);
            arma::sp_mat::const_iterator it = S.begin();
            arma::sp_mat::const_iterator it_end = S.end();
            for (; it != it_end; ++it) {
                int i = it.row();
                int j = sample_assignments[it.col()] - 1;
                pb(i, j) += (*it);
            }
            return pb;
        } else { // columnwise
            arma::mat pb = arma::zeros(lv_vec.n_elem, S.n_cols);
            arma::sp_mat::const_iterator it = S.begin();
            arma::sp_mat::const_iterator it_end = S.end();
            for (; it != it_end; ++it) {
                int i = sample_assignments[it.row()] - 1;
                int j = it.col();
                pb(i, j) += (*it);
            }
            return pb;
        }
    }

    arma::mat computeGroupedSums(arma::mat& S, arma::vec& sample_assignments, int axis) {
        if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
            throw std::invalid_argument("Length of 'sample_assignments' must match the number of columns (axis=0) or rows (axis=1) of S.");
        }
        arma::vec lv_vec = arma::unique(sample_assignments);
        if (axis == 0) { // rowwise (default)
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
            return pb;
        } else { // columnwise
            arma::mat pb = arma::zeros(lv_vec.n_elem, S.n_cols);
            for (int i = 0; i < pb.n_rows; i++) {
                arma::uvec idx = arma::find(sample_assignments == (i + 1));
                if (idx.n_elem == 0) continue;
                if (idx.n_elem > 1) {
                    arma::mat subS = S.rows(idx);
                    pb.row(i) = arma::sum(subS, 0);
                } else {
                    pb.row(i) = S.row(idx(0));
                }
            }
            return pb;
        }
    }

    arma::sp_mat computeGroupedSums2(arma::sp_mat& S, arma::vec& sample_assignments, int axis) {
        if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
            throw std::invalid_argument("Length of sample_assignments must match the number of columns (axis=0) or rows (axis=1) of S.");
        }
        arma::vec lv_vec = arma::unique(sample_assignments);
        if (axis == 0) {
            arma::sp_mat pb(S.n_rows, lv_vec.n_elem);
            arma::sp_mat::const_iterator it = S.begin();
            arma::sp_mat::const_iterator it_end = S.end();
            for (; it != it_end; ++it) {
                int i = it.row();
                int j = sample_assignments[it.col()] - 1;
                pb(i, j) += (*it);
            }
            return pb;
        } else {
            arma::sp_mat pb(lv_vec.n_elem, S.n_cols);
            arma::sp_mat::const_iterator it = S.begin();
            arma::sp_mat::const_iterator it_end = S.end();
            for (; it != it_end; ++it) {
                int i = sample_assignments[it.row()] - 1;
                int j = it.col();
                pb(i, j) += (*it);
            }
            return pb;
        }
    }

    template <typename T>
    arma::mat computeGroupedMeans(T& S, arma::vec& sample_assignments, int axis) {
        if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
            throw std::invalid_argument("Length of 'sample_assignments' must match the number of columns (axis=0) or rows (axis=1) of S.");
        }
        arma::mat pb = computeGroupedSums(S, sample_assignments, axis);
        if (axis == 0) {
            for (int j = 0; j < pb.n_cols; j++) {
                arma::uvec idx = arma::find(sample_assignments == (j + 1));
                pb.col(j) /= std::max(1, (int)idx.n_elem);
            }
        } else {
            for (int i = 0; i < pb.n_rows; i++) {
                arma::uvec idx = arma::find(sample_assignments == (i + 1));
                pb.row(i) /= std::max(1, (int)idx.n_elem);
            }
        }
        return pb;
    }

    template arma::mat computeGroupedMeans<arma::mat>(arma::mat& S, arma::vec& sample_assignments, int axis);
    template arma::mat computeGroupedMeans<arma::sp_mat>(arma::sp_mat& S, arma::vec& sample_assignments, int axis);

    arma::sp_mat computeGroupedMeans2(arma::sp_mat& S, arma::vec& sample_assignments, int axis) {
        if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
            throw std::invalid_argument("Length of sample_assignments must match the number of columns (axis=0) or rows (axis=1) of S.");
        }
        arma::vec lv_vec = arma::unique(sample_assignments);
        if (axis == 0) {
            arma::sp_mat pb(S.n_rows, lv_vec.n_elem);
            arma::sp_mat pbz(S.n_rows, lv_vec.n_elem);
            arma::sp_mat::const_iterator it = S.begin();
            arma::sp_mat::const_iterator it_end = S.end();
            for (; it != it_end; ++it) {
                int i = it.row();
                int j = sample_assignments[it.col()] - 1;
                pb(i, j) += (*it);
                pbz(i, j) += 1;
            }
            for (int j = 0; j < pb.n_cols; ++j) {
                arma::uvec idx = arma::find(sample_assignments == (j + 1));
                int nnz = (int)idx.n_elem;
                double denom = std::max(1, nnz);
                for (int i = 0; i < pb.n_rows; ++i) {
                    if (pb(i, j) != 0.0) pb(i, j) /= denom;
                }
            }
            return pb;
        } else {
            arma::sp_mat pb(lv_vec.n_elem, S.n_cols);
            arma::sp_mat pbz(lv_vec.n_elem, S.n_cols);
            arma::sp_mat::const_iterator it = S.begin();
            arma::sp_mat::const_iterator it_end = S.end();
            for (; it != it_end; ++it) {
                int i = sample_assignments[it.row()] - 1;
                int j = it.col();
                pb(i, j) += (*it);
                pbz(i, j) += 1;
            }
            for (int i = 0; i < pb.n_rows; i++) {
                arma::uvec idx = arma::find(sample_assignments == (i + 1));
                int nnz = (int)idx.n_elem;
                double denom = std::max(1, nnz);
                for (int j = 0; j < pb.n_cols; ++j) {
                    if (pb(i, j) != 0.0) pb(i, j) /= denom;
                }
            }
            return pb;
        }
    }

    arma::mat computeGroupedVars(arma::sp_mat& S, arma::vec& sample_assignments, int axis) {
        if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
            throw std::invalid_argument("Length of 'sample_assignments' must match the number of columns (axis=0) or rows (axis=1) of S.");
        }
        arma::mat pb_mu = computeGroupedMeans(S, sample_assignments, axis);
        if (axis == 0) {
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
                int nnz = (int)idx.n_elem;
                for (int i = 0; i < pb.n_rows; i++) {
                    int nz = (int)idx.n_elem - pbz(i, j);
                    pb(i, j) += nz * pb_mu(i, j) * pb_mu(i, j);
                }
                pb.col(j) /= std::max(1, nnz - 1);
            }
            return pb;
        } else {
            arma::mat pb = arma::zeros(pb_mu.n_rows, pb_mu.n_cols);
            arma::mat pbz = arma::zeros(pb_mu.n_rows, pb_mu.n_cols);
            arma::sp_mat::const_iterator it = S.begin();
            arma::sp_mat::const_iterator it_end = S.end();
            for (; it != it_end; ++it) {
                int i = sample_assignments[it.row()] - 1;
                int j = it.col();
                double num = (*it) - pb_mu(i, j);
                pb(i, j) += num * num;
                pbz(i, j) += 1;
            }
            for (int i = 0; i < pb.n_rows; i++) {
                arma::uvec idx = arma::find(sample_assignments == (i + 1));
                int nnz = (int)idx.n_elem;
                for (int j = 0; j < pb.n_cols; j++) {
                    int nz = (int)idx.n_elem - pbz(i, j);
                    pb(i, j) += nz * pb_mu(i, j) * pb_mu(i, j);
                }
                pb.row(i) /= std::max(1, nnz - 1);
            }
            return pb;
        }
    }

    arma::mat computeGroupedVars(arma::mat& S, arma::vec& sample_assignments, int axis) {
        if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
            throw std::invalid_argument("Length of 'sample_assignments' must match the number of columns (axis=0) or rows (axis=1) of S.");
        }
        arma::vec lv_vec = arma::unique(sample_assignments);
        if (axis == 0) {
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
            return pb;
        } else {
            arma::mat pb = arma::zeros(lv_vec.n_elem, S.n_cols);
            for (int i = 0; i < pb.n_rows; i++) {
                arma::uvec idx = arma::find(sample_assignments == (i + 1));
                if (idx.n_elem == 0) continue;
                if (idx.n_elem > 1) {
                    arma::mat subS = S.rows(idx);
                    pb.row(i) = arma::var(subS, 0, 0);
                } else {
                    pb.row(i) = arma::zeros(pb.n_cols);
                }
            }
            return pb;
        }
    }

arma::sp_mat computeGroupedVars2(arma::sp_mat& S, arma::vec& sample_assignments, int axis) {
    if ((axis == 0 && sample_assignments.n_elem != S.n_cols) || (axis == 1 && sample_assignments.n_elem != S.n_rows)) {
        throw std::invalid_argument("Length of sample_assignments must match the number of columns (axis=0) or rows (axis=1) of S.");
    }
    arma::vec lv_vec = arma::unique(sample_assignments);
    if (axis == 0) {
        arma::sp_mat mean_pb = computeGroupedMeans2(S, sample_assignments, axis);
        arma::sp_mat pb(S.n_rows, lv_vec.n_elem);
        arma::sp_mat pbz(S.n_rows, lv_vec.n_elem);
        arma::sp_mat::const_iterator it = S.begin();
        arma::sp_mat::const_iterator it_end = S.end();
        for (; it != it_end; ++it) {
            int i = it.row();
            int j = sample_assignments[it.col()] - 1;
            double num = (*it) - mean_pb(i, j);
            pb(i, j) += num * num;
            pbz(i, j) += 1;
        }
        for (int j = 0; j < pb.n_cols; ++j) {
            arma::uvec idx = arma::find(sample_assignments == (j + 1));
            int nnz = (int)idx.n_elem;
            for (int i = 0; i < pb.n_rows; ++i) {
                int nz = nnz - pbz(i, j);
                pb(i, j) += nz * mean_pb(i, j) * mean_pb(i, j);
            }
            double denom = std::max(1, nnz - 1);
            for (int i = 0; i < pb.n_rows; ++i) {
                if (pb(i, j) != 0.0) pb(i, j) /= denom;
            }
        }
        return pb;
    } else {
        arma::sp_mat mean_pb = computeGroupedMeans2(S, sample_assignments, axis);
        arma::sp_mat pb(lv_vec.n_elem, S.n_cols);
        arma::sp_mat pbz(lv_vec.n_elem, S.n_cols);
        arma::sp_mat::const_iterator it = S.begin();
        arma::sp_mat::const_iterator it_end = S.end();
        for (; it != it_end; ++it) {
            int i = sample_assignments[it.row()] - 1;
            int j = it.col();
            double num = (*it) - mean_pb(i, j);
            pb(i, j) += num * num;
            pbz(i, j) += 1;
        }
        for (int i = 0; i < pb.n_rows; ++i) {
            arma::uvec idx = arma::find(sample_assignments == (i + 1));
            int nnz = (int)idx.n_elem;
            for (int j = 0; j < pb.n_cols; ++j) {
                int nz = nnz - pbz(i, j);
                pb(i, j) += nz * mean_pb(i, j) * mean_pb(i, j);
            }
            double denom = std::max(1, nnz - 1);
            for (int j = 0; j < pb.n_cols; ++j) {
                if (pb(i, j) != 0.0) pb(i, j) /= denom;
            }
        }
        return pb;
    }
}

} // namespace actionet
