#include "annotation/specificity.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "utils_internal/utils_matrix.hpp"

namespace {

// Shared Bernstein tail-bound computation used by all paths.
// Inputs:
//   Obs        — genes x k observed co-occurrence
//   row_p      — genes-length gene density
//   row_factor — genes-length mean of nonzero per gene
//   col_p      — cells-length cell density
//   H_norm     — cells x k normalised membership matrix
//   n_obs      — number of observations (cells)
// Outputs via res field(3): average_profile, upper_significance, lower_significance
arma::field<arma::mat> bernstein_tail_bounds(
        const arma::mat& Obs,
        const arma::vec& row_p,
        const arma::vec& row_factor,
        const arma::vec& col_p,
        const arma::mat& H_norm,
        arma::uword n_obs) {

    const arma::uword k = H_norm.n_cols;
    const double rho = arma::mean(col_p);
    arma::vec beta = (rho == 0.0) ? arma::zeros(n_obs) : arma::vec(col_p / rho);

    arma::mat Gamma = H_norm;
    arma::vec a(k);
    for (arma::uword j = 0; j < k; ++j) {
        Gamma.col(j) %= beta;
        a(j) = arma::max(Gamma.col(j));
    }

    arma::mat Exp    = (row_p % row_factor) * arma::sum(Gamma, 0);
    arma::mat Nu     = (row_p % arma::square(row_factor)) * arma::sum(arma::square(Gamma), 0);
    arma::mat A      = row_factor * arma::trans(a);
    arma::mat Lambda = Obs - Exp;

    arma::mat logPvals_lower = arma::square(Lambda) / (2.0 * Nu);
    logPvals_lower(arma::find(Lambda >= 0)).zeros();
    logPvals_lower.replace(arma::datum::nan, 0.0);

    arma::mat logPvals_upper = arma::square(Lambda) / (2.0 * (Nu + (Lambda % A / 3.0)));
    logPvals_upper(arma::find(Lambda <= 0)).zeros();
    logPvals_upper.replace(arma::datum::nan, 0.0);

    const double log10_e = std::log(10.0);
    logPvals_lower /= log10_e;
    logPvals_upper /= log10_e;

    arma::field<arma::mat> res(3);
    res(0) = Obs / static_cast<double>(n_obs);
    res(1) = logPvals_upper;
    res(2) = logPvals_lower;
    return res;
}

// Normalise H column-wise: divide each column by its mean.
arma::mat normalise_H(const arma::mat& H) {
    arma::mat H_norm = H;
    for (arma::uword j = 0; j < H_norm.n_cols; ++j) {
        double mu = arma::mean(H_norm.col(j));
        if (mu != 0.0) H_norm.col(j) /= mu;
    }
    return H_norm;
}

} // anon namespace

// Dense getProbsObs: no full copy for binarization — count > 0 directly.
static void getProbsObs_dense(const arma::mat& S, const arma::mat& Ht,
                               arma::vec& row_factor, arma::vec& row_p,
                               arma::vec& col_p, arma::mat& Obs) {
    const arma::uword n_rows = S.n_rows;
    const arma::uword n_cols = S.n_cols;

    arma::vec gene_nnz  = arma::zeros(n_cols);
    arma::vec gene_sum  = arma::zeros(n_cols);
    arma::vec cell_nnz  = arma::zeros(n_rows);

    for (arma::uword c = 0; c < n_cols; ++c) {
        for (arma::uword r = 0; r < n_rows; ++r) {
            double v = S(r, c);
            if (v > 0.0) {
                gene_nnz(c) += 1.0;
                cell_nnz(r) += 1.0;
            }
            gene_sum(c) += v;
        }
    }

    row_factor = gene_sum / gene_nnz;
    row_factor.replace(arma::datum::nan, 0.0);
    row_p = gene_nnz / n_rows;
    col_p = cell_nnz / n_cols;
    Obs = S.t() * Ht;
}

// Sparse getProbsObs: iterate nonzeros only (no copy, no densification).
// Also accumulates support_obs(g, j) = sum of Ht(r, j) over stored nonzeros
// for gene g, needed for per-gene analytical shift correction.
static void getProbsObs_sparse(const arma::sp_mat& S, const arma::mat& Ht, int thread_no,
                                arma::vec& row_factor, arma::vec& row_p,
                                arma::vec& col_p, arma::mat& Obs,
                                arma::mat& support_obs) {
    const arma::uword n_rows = S.n_rows;
    const arma::uword n_cols = S.n_cols;
    const arma::uword k = Ht.n_cols;

    arma::vec gene_nnz = arma::zeros(n_cols);
    arma::vec cell_nnz = arma::zeros(n_rows);
    arma::vec gene_sum = arma::zeros(n_cols);
    support_obs.zeros(n_cols, k);

    for (auto it = S.begin(); it != S.end(); ++it) {
        double v = (*it);
        arma::uword r = it.row();
        arma::uword c = it.col();
        if (v > 0.0) {
            gene_nnz(c) += 1.0;
            cell_nnz(r) += 1.0;
        }
        gene_sum(c) += v;
        support_obs.row(c) += Ht.row(r);
    }

    row_factor = gene_sum / gene_nnz;
    row_factor.replace(arma::datum::nan, 0.0);
    row_p = gene_nnz / n_rows;
    col_p = cell_nnz / n_cols;
    Obs = spmat_mat_product_parallel(S.t(), Ht, thread_no);
}

namespace actionet {

    // In-memory dense path
    template <>
    arma::field<arma::mat> computeFeatureSpecificity<arma::mat>(
            const arma::mat& S, const arma::mat& H, int thread_no) {
        stdout_printf("Computing feature specificity ... ");

        double min_val = S.min();
        arma::mat H_norm = normalise_H(H);

        arma::vec row_factor, row_p, col_p;
        arma::mat Obs;

        if (min_val == 0.0) {
            getProbsObs_dense(S, H_norm, row_factor, row_p, col_p, Obs);
        } else {
            arma::mat S_shifted = S;
            S_shifted -= min_val;
            getProbsObs_dense(S_shifted, H_norm, row_factor, row_p, col_p, Obs);
        }

        stdout_printf("done\n");
        FLUSH;
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, S.n_rows);
    }

    // In-memory sparse path: no copy, no densification.
    // Apply shift analytically using per-gene support_obs (matching backed sparse path).
    template <>
    arma::field<arma::mat> computeFeatureSpecificity<arma::sp_mat>(
            const arma::sp_mat& S, const arma::mat& H, int thread_no) {
        stdout_printf("Computing feature specificity ... ");

        double min_val = S.min();
        arma::mat H_norm = normalise_H(H);

        arma::vec row_factor, row_p, col_p;
        arma::mat Obs, support_obs;

        getProbsObs_sparse(S, H_norm, thread_no, row_factor, row_p, col_p, Obs, support_obs);

        if (min_val < 0.0) {
            double shift = -min_val;
            Obs += shift * support_obs;
            row_factor += shift;
            row_p.ones();
            col_p.ones();
        }

        stdout_printf("done\n");
        FLUSH;
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, S.n_rows);
    }

    // Label-based overloads: build H from labels and delegate.
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(const T& S, const arma::uvec& labels, int thread_no) {
        arma::mat H(S.n_rows, arma::max(labels), arma::fill::zeros);
        for (arma::uword i = 1; i <= arma::max(labels); i++) {
            arma::uvec idx = arma::find(labels == i);
            for (arma::uword j : idx) {
                H(j, i - 1) = 1.0;
            }
        }
        return computeFeatureSpecificity(S, H, thread_no);
    }

    template arma::field<arma::mat> computeFeatureSpecificity<arma::mat>(
        const arma::mat& S, const arma::uvec& labels, int thread_no);
    template arma::field<arma::mat> computeFeatureSpecificity<arma::sp_mat>(
        const arma::sp_mat& S, const arma::uvec& labels, int thread_no);

} // namespace actionet

// ============================================================================
// Backed sparse overloads
// ============================================================================

namespace actionet {

    void backed_specificity_scan_csr_(
        const BackedSparseMatrixOperator& op,
        const arma::mat& H_norm_t,
        arma::vec& row_count,
        arma::vec& col_count,
        arma::vec& row_factor_sum_orig,
        arma::mat& obs_orig,
        arma::mat& support_obs,
        double& min_stored)
    {
        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;
        const arma::uword cs    = op.chunk_size_;

        row_count.zeros(n_var);
        col_count.zeros(n_obs);
        row_factor_sum_orig.zeros(n_var);
        obs_orig.zeros(n_var, H_norm_t.n_cols);
        support_obs.zeros(n_var, H_norm_t.n_cols);
        min_stored = 0.0;

        const std::vector<double>* data;
        const std::vector<unsigned long long>* indices;

        for (arma::uword row_start = 0; row_start < n_obs; row_start += cs) {
            const arma::uword row_end = std::min<arma::uword>(n_obs, row_start + cs);
            const unsigned long long nnz_start = op.indptr_[row_start];
            const unsigned long long nnz_end   = op.indptr_[row_end];
            const unsigned long long nnz_count  = nnz_end - nnz_start;

            op.load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long p0 = op.indptr_[r]     - nnz_start;
                const unsigned long long p1 = op.indptr_[r + 1] - nnz_start;

                const arma::rowvec h_row = H_norm_t.row(r);

                for (unsigned long long p = p0; p < p1; ++p) {
                    const arma::uword c = static_cast<arma::uword>(
                        (*indices)[static_cast<size_t>(p)]);
                    const double v = op.transform_value_(r, (*data)[static_cast<size_t>(p)]);

                    if (v < min_stored) min_stored = v;
                    if (v > 0.0) {
                        row_count(c) += 1.0;
                        col_count(r) += 1.0;
                    }
                    row_factor_sum_orig(c) += v;
                    obs_orig.row(c)        += v * h_row;
                    support_obs.row(c)     += h_row;
                }
            }
        }
    }

    void backed_specificity_scan_csc_(
        const BackedSparseMatrixOperator& op,
        const arma::mat& H_norm_t,
        arma::vec& row_count,
        arma::vec& col_count,
        arma::vec& row_factor_sum_orig,
        arma::mat& obs_orig,
        arma::mat& support_obs,
        double& min_stored)
    {
        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;
        const arma::uword cs    = op.chunk_size_;

        row_count.zeros(n_var);
        col_count.zeros(n_obs);
        row_factor_sum_orig.zeros(n_var);
        obs_orig.zeros(n_var, H_norm_t.n_cols);
        support_obs.zeros(n_var, H_norm_t.n_cols);
        min_stored = 0.0;

        const std::vector<double>* data;
        const std::vector<unsigned long long>* indices;

        for (arma::uword col_start = 0; col_start < n_var; col_start += cs) {
            const arma::uword col_end = std::min<arma::uword>(n_var, col_start + cs);
            const unsigned long long nnz_start = op.indptr_[col_start];
            const unsigned long long nnz_end   = op.indptr_[col_end];
            const unsigned long long nnz_count  = nnz_end - nnz_start;

            op.load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long p0 = op.indptr_[c]     - nnz_start;
                const unsigned long long p1 = op.indptr_[c + 1] - nnz_start;

                for (unsigned long long p = p0; p < p1; ++p) {
                    const arma::uword r = static_cast<arma::uword>(
                        (*indices)[static_cast<size_t>(p)]);
                    const double v = op.transform_value_(r, (*data)[static_cast<size_t>(p)]);

                    if (v < min_stored) min_stored = v;
                    if (v > 0.0) {
                        row_count(c) += 1.0;
                        col_count(r) += 1.0;
                    }
                    row_factor_sum_orig(c) += v;

                    const arma::rowvec h_row = H_norm_t.row(r);
                    obs_orig.row(c)    += v * h_row;
                    support_obs.row(c) += h_row;
                }
            }
        }
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::mat& H, int /*thread_no*/) {
        stdout_printf("Computing feature specificity (backed sparse) ... ");

        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;

        arma::mat H_norm = normalise_H(H);

        arma::vec row_count, col_count, row_factor_sum_orig;
        arma::mat obs_orig, support_obs;
        double min_stored;

        if (op.is_csr_) {
            backed_specificity_scan_csr_(op, H_norm, row_count, col_count,
                                         row_factor_sum_orig, obs_orig, support_obs, min_stored);
        } else {
            backed_specificity_scan_csc_(op, H_norm, row_count, col_count,
                                         row_factor_sum_orig, obs_orig, support_obs, min_stored);
        }

        const double shift = (min_stored < 0.0) ? -min_stored : 0.0;
        arma::vec row_factor_sum = row_factor_sum_orig + shift * row_count;
        arma::mat Obs = obs_orig + shift * support_obs;

        arma::vec row_factor = arma::zeros(n_var);
        for (arma::uword i = 0; i < n_var; ++i) {
            if (row_count(i) > 0.0) row_factor(i) = row_factor_sum(i) / row_count(i);
        }
        arma::vec row_p = row_count / static_cast<double>(n_obs);
        arma::vec col_p = col_count / static_cast<double>(n_var);

        stdout_printf("done\n");
        FLUSH;
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, n_obs);
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no) {
        const arma::uword max_label = arma::max(labels);
        arma::mat H(op.n_obs_, max_label, arma::fill::zeros);
        for (arma::uword i = 1; i <= max_label; ++i) {
            arma::uvec idx = arma::find(labels == i);
            for (arma::uword j : idx) H(j, i - 1) = 1.0;
        }
        return computeFeatureSpecificity(op, H, thread_no);
    }

} // namespace actionet

// ============================================================================
// Backed dense overloads
// ============================================================================
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

namespace actionet {

    static void backed_dense_specificity_scan_(
        BackedDenseMatrixOperator& op,
        arma::vec& row_count,
        arma::vec& col_count,
        arma::vec& row_factor_sum,
        double&    global_min)
    {
        const arma::uword n_obs = op.rows();
        const arma::uword n_var = op.cols();
        const arma::uword cs    = op.effectiveChunkSize();

        row_factor_sum.zeros(n_var);
        global_min = std::numeric_limits<double>::infinity();

        arma::mat slab;
        for (arma::uword obs_start = 0; obs_start < n_obs; obs_start += cs) {
            const arma::uword obs_count = std::min(cs, n_obs - obs_start);
            op.readSlab(obs_start, obs_count, slab);
            const double chunk_min = slab.min();
            if (chunk_min < global_min) global_min = chunk_min;
        }
        if (!std::isfinite(global_min)) global_min = 0.0;

        row_count.zeros(n_var);
        col_count.zeros(n_obs);

        for (arma::uword obs_start = 0; obs_start < n_obs; obs_start += cs) {
            const arma::uword obs_count = std::min(cs, n_obs - obs_start);
            op.readSlab(obs_start, obs_count, slab);

            if (global_min != 0.0) slab -= global_min;

            row_factor_sum += arma::sum(slab, 0).t();

            for (arma::uword r = 0; r < obs_count; ++r) {
                for (arma::uword c = 0; c < n_var; ++c) {
                    if (slab(r, c) > 0.0) {
                        row_count(c) += 1.0;
                        col_count(obs_start + r) += 1.0;
                    }
                }
            }
        }
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedDenseMatrixOperator& op,
                                                     const arma::mat& H, int thread_no) {
        stdout_printf("Computing feature specificity (backed dense) ... ");

        const arma::uword n_obs = op.rows();
        const arma::uword n_var = op.cols();

        arma::mat H_norm = normalise_H(H);

        arma::vec row_count, col_count, row_factor_sum;
        double global_min;
        backed_dense_specificity_scan_(op, row_count, col_count, row_factor_sum, global_min);

        arma::mat Obs;
        op.rmatmat(H_norm, Obs);

        if (global_min != 0.0) {
            arma::rowvec H_norm_colsums = arma::sum(H_norm, 0);
            Obs -= global_min * arma::ones(n_var, 1) * H_norm_colsums;
        }

        arma::vec row_factor = arma::zeros(n_var);
        for (arma::uword i = 0; i < n_var; ++i) {
            if (row_count(i) > 0.0) row_factor(i) = row_factor_sum(i) / row_count(i);
        }
        arma::vec row_p = row_count / static_cast<double>(n_obs);
        arma::vec col_p = col_count / static_cast<double>(n_var);

        stdout_printf("done\n");
        FLUSH;
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, n_obs);
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedDenseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no) {
        const arma::uword max_label = arma::max(labels);
        arma::mat H(op.rows(), max_label, arma::fill::zeros);
        for (arma::uword i = 1; i <= max_label; ++i) {
            arma::uvec idx = arma::find(labels == i);
            for (arma::uword j : idx) H(j, i - 1) = 1.0;
        }
        return computeFeatureSpecificity(op, H, thread_no);
    }

} // namespace actionet
