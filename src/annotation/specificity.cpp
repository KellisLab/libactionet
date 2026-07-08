#include "annotation/specificity.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"
#include "utils_internal/utils_matrix.hpp"
#include "utils_internal/utils_parallel.hpp"
#include <algorithm>
#include <cmath>

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
        arma::uword n_obs,
        int thread_no) {

    const arma::uword k = H_norm.n_cols;
    const double rho = arma::mean(col_p);
    const double log10_e = std::log(10.0);
    arma::vec beta = (rho == 0.0) ? arma::zeros(n_obs) : arma::vec(col_p / rho);

    // Compute per-group Gamma summary statistics without materialising Gamma.
    // This avoids an additional n_obs x k dense matrix allocation.
    arma::rowvec gamma_sum(k, arma::fill::zeros);
    arma::rowvec gamma_sq_sum(k, arma::fill::zeros);
    arma::rowvec a(k, arma::fill::zeros);

    const unsigned int threads_use_gamma = actionet::get_num_threads(
        static_cast<unsigned int>(k), static_cast<unsigned int>(std::max(thread_no, 0)));
    #pragma omp parallel for schedule(static) num_threads(threads_use_gamma) if(threads_use_gamma > 1 && k > 1)
    for (arma::sword js = 0; js < static_cast<arma::sword>(k); ++js) {
        const arma::uword j = static_cast<arma::uword>(js);
        const arma::vec h_col = H_norm.col(j);
        const arma::vec g_col = h_col % beta;
        gamma_sum(j) = arma::accu(g_col);
        gamma_sq_sum(j) = arma::dot(g_col, g_col);
        a(j) = g_col.max();
    }

    const arma::vec rp_rf = row_p % row_factor;
    const arma::vec rp_rf2 = row_p % arma::square(row_factor);

    arma::mat logPvals_upper(Obs.n_rows, Obs.n_cols, arma::fill::zeros);
    arma::mat logPvals_lower(Obs.n_rows, Obs.n_cols, arma::fill::zeros);

    const unsigned int threads_use_tail = actionet::get_num_threads(
        static_cast<unsigned int>(k), static_cast<unsigned int>(std::max(thread_no, 0)));
    #pragma omp parallel for schedule(static) num_threads(threads_use_tail) if(threads_use_tail > 1 && k > 1)
    for (arma::sword js = 0; js < static_cast<arma::sword>(k); ++js) {
        const arma::uword j = static_cast<arma::uword>(js);
        const double gs = gamma_sum(j);
        const double gs2 = gamma_sq_sum(j);
        const double aj = a(j);

        for (arma::uword i = 0; i < Obs.n_rows; ++i) {
            const double exp_ij = rp_rf(i) * gs;
            const double nu_ij = rp_rf2(i) * gs2;
            const double lambda_ij = Obs(i, j) - exp_ij;

            if (lambda_ij < 0.0 && nu_ij > 0.0) {
                const double lower = (lambda_ij * lambda_ij) / (2.0 * nu_ij);
                if (std::isfinite(lower)) {
                    logPvals_lower(i, j) = lower / log10_e;
                }
            }

            if (lambda_ij > 0.0) {
                const double A_ij = row_factor(i) * aj;
                const double denom = 2.0 * (nu_ij + (lambda_ij * A_ij / 3.0));
                if (denom > 0.0) {
                    const double upper = (lambda_ij * lambda_ij) / denom;
                    if (std::isfinite(upper)) {
                        logPvals_upper(i, j) = upper / log10_e;
                    }
                }
            }
        }
    }

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
// Optionally accumulates support_obs(g, j) = sum of Ht(r, j) over stored
// nonzeros for gene g, needed only when min-shift correction is required.
static void getProbsObs_sparse(const arma::sp_mat& S, const arma::mat& Ht, int thread_no,
                                arma::vec& row_factor, arma::vec& row_p,
                                arma::vec& col_p, arma::mat& Obs,
                                arma::mat* support_obs) {
    const arma::uword n_rows = S.n_rows;
    const arma::uword n_cols = S.n_cols;
    const arma::uword k = Ht.n_cols;

    arma::vec gene_nnz = arma::zeros(n_cols);
    arma::vec cell_nnz = arma::zeros(n_rows);
    arma::vec gene_sum = arma::zeros(n_cols);
    if (support_obs != nullptr) {
        support_obs->zeros(n_cols, k);
    }

    for (auto it = S.begin(); it != S.end(); ++it) {
        double v = (*it);
        arma::uword r = it.row();
        arma::uword c = it.col();
        if (v > 0.0) {
            gene_nnz(c) += 1.0;
            cell_nnz(r) += 1.0;
        }
        gene_sum(c) += v;
        if (support_obs != nullptr) {
            support_obs->row(c) += Ht.row(r);
        }
    }

    row_factor = gene_sum / gene_nnz;
    row_factor.replace(arma::datum::nan, 0.0);
    row_p = gene_nnz / n_rows;
    col_p = cell_nnz / n_cols;
    Obs = actionet::spmat_mat_product_parallel(S.t(), Ht, thread_no);
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

        getProbsObs_dense(S, H_norm, row_factor, row_p, col_p, Obs);

        // Apply min-shift analytically: for a dense matrix every cell contributes
        // to every gene's support, so support_obs == sum(H_norm, 0) broadcast over
        // genes.  This avoids a full O(n_cells * n_genes) matrix copy.
        if (min_val != 0.0) {
            const double shift = -min_val;
            // Obs correction: each gene gets shift * (sum of H_norm rows) = shift * H_norm_colsums
            arma::rowvec H_norm_colsums = arma::sum(H_norm, 0);
            Obs += shift * arma::ones(S.n_cols, 1) * H_norm_colsums;
            // row_factor = mean nonzero value; after shift all values > 0, so
            // row_factor_new = (gene_sum + shift * n_rows) / n_rows = old_mean + shift
            // (gene_nnz becomes n_rows and gene_sum increases by shift * n_rows)
            row_factor += shift;
            row_p.ones();
            col_p.ones();
        }

        stdout_printf("done\n");
        FLUSH;
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, S.n_rows, thread_no);
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
        arma::mat* support_obs_ptr = nullptr;

        if (min_val < 0.0) {
            support_obs_ptr = &support_obs;
        }

        getProbsObs_sparse(S, H_norm, thread_no, row_factor, row_p, col_p, Obs, support_obs_ptr);

        if (min_val < 0.0) {
            double shift = -min_val;
            Obs += shift * support_obs;
            row_factor += shift;
            row_p.ones();
            col_p.ones();
        }

        stdout_printf("done\n");
        FLUSH;
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, S.n_rows, thread_no);
    }

    // Label-based overloads: build H from labels and delegate.
    // Single O(n_cells) scatter: avoids k full scans via arma::find.
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(const T& S, const arma::uvec& labels, int thread_no) {
        const arma::uword max_label = arma::max(labels);
        arma::mat H(S.n_rows, max_label, arma::fill::zeros);
        for (arma::uword j = 0; j < S.n_rows; ++j) {
            if (labels(j) > 0) H(j, labels(j) - 1) = 1.0;
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
        double& min_stored,
        int thread_no)
    {
        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;
        const arma::uword k     = H_norm_t.n_cols;
        const arma::uword cs    = op.chunk_size_;
        const unsigned int threads_use_obs = actionet::get_num_threads(
            static_cast<unsigned int>(k), static_cast<unsigned int>(std::max(thread_no, 0)));

        row_count.zeros(n_var);
        col_count.zeros(n_obs);
        row_factor_sum_orig.zeros(n_var);
        obs_orig.zeros(n_var, k);
        min_stored = 0.0;

        const std::vector<double>* data;
        const std::vector<unsigned long long>* indices;

        for (arma::uword row_start = 0; row_start < n_obs; row_start += cs) {
            const arma::uword row_end = std::min<arma::uword>(n_obs, row_start + cs);
            const unsigned long long nnz_start = op.indptr_[row_start];
            const unsigned long long nnz_end   = op.indptr_[row_end];
            const unsigned long long nnz_count  = nnz_end - nnz_start;

            op.load_chunk_cached_(nnz_start, nnz_count, data, indices);
            op.ensure_chunk_transformed_csr_(row_start, row_end, nnz_start);

            // Pass 1: scalar scan for support counts/sums and min.
            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long p0 = op.indptr_[r]     - nnz_start;
                const unsigned long long p1 = op.indptr_[r + 1] - nnz_start;
                double row_pos = 0.0;

                for (unsigned long long p = p0; p < p1; ++p) {
                    const arma::uword c = static_cast<arma::uword>(
                        (*indices)[static_cast<size_t>(p)]);
                    const double v = (*data)[static_cast<size_t>(p)];

                    if (v < min_stored) min_stored = v;
                    if (v > 0.0) {
                        row_count(c) += 1.0;
                        row_pos += 1.0;
                    }
                    row_factor_sum_orig(c) += v;
                }
                col_count(r) = row_pos;
            }

            // Pass 2: Obs accumulation parallelized across specificity columns.
            if (threads_use_obs > 1 && k > 1) {
                #pragma omp parallel for schedule(static) num_threads(threads_use_obs)
                for (arma::sword js = 0; js < static_cast<arma::sword>(k); ++js) {
                    const arma::uword j = static_cast<arma::uword>(js);
                    const double* h_col = H_norm_t.colptr(j);
                    double* obs_col = obs_orig.colptr(j);

                    for (arma::uword r = row_start; r < row_end; ++r) {
                        const double h = h_col[r];
                        if (h == 0.0) continue;

                        const unsigned long long p0 = op.indptr_[r]     - nnz_start;
                        const unsigned long long p1 = op.indptr_[r + 1] - nnz_start;
                        for (unsigned long long p = p0; p < p1; ++p) {
                            const arma::uword c = static_cast<arma::uword>(
                                (*indices)[static_cast<size_t>(p)]);
                            obs_col[c] += (*data)[static_cast<size_t>(p)] * h;
                        }
                    }
                }
            } else {
                for (arma::uword r = row_start; r < row_end; ++r) {
                    const unsigned long long p0 = op.indptr_[r]     - nnz_start;
                    const unsigned long long p1 = op.indptr_[r + 1] - nnz_start;

                    for (unsigned long long p = p0; p < p1; ++p) {
                        const arma::uword c = static_cast<arma::uword>(
                            (*indices)[static_cast<size_t>(p)]);
                        const double v = (*data)[static_cast<size_t>(p)];
                        double* obs_base = obs_orig.memptr() + c;
                        for (arma::uword j = 0; j < k; ++j) {
                            obs_base[j * n_var] += v * H_norm_t(r, j);
                        }
                    }
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
        double& min_stored,
        int thread_no)
    {
        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;
        const arma::uword k     = H_norm_t.n_cols;
        const arma::uword cs    = op.chunk_size_;
        const unsigned int threads_use_obs = actionet::get_num_threads(
            static_cast<unsigned int>(k), static_cast<unsigned int>(std::max(thread_no, 0)));

        row_count.zeros(n_var);
        col_count.zeros(n_obs);
        row_factor_sum_orig.zeros(n_var);
        obs_orig.zeros(n_var, k);
        min_stored = 0.0;

        const std::vector<double>* data;
        const std::vector<unsigned long long>* indices;

        for (arma::uword col_start = 0; col_start < n_var; col_start += cs) {
            const arma::uword col_end = std::min<arma::uword>(n_var, col_start + cs);
            const unsigned long long nnz_start = op.indptr_[col_start];
            const unsigned long long nnz_end   = op.indptr_[col_end];
            const unsigned long long nnz_count  = nnz_end - nnz_start;

            op.load_chunk_cached_(nnz_start, nnz_count, data, indices);
            op.ensure_chunk_transformed_csc_();

            // Pass 1: scalar scan for support counts/sums and min.
            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long p0 = op.indptr_[c]     - nnz_start;
                const unsigned long long p1 = op.indptr_[c + 1] - nnz_start;

                for (unsigned long long p = p0; p < p1; ++p) {
                    const arma::uword r = static_cast<arma::uword>(
                        (*indices)[static_cast<size_t>(p)]);
                    const double v = (*data)[static_cast<size_t>(p)];

                    if (v < min_stored) min_stored = v;
                    if (v > 0.0) {
                        row_count(c) += 1.0;
                        col_count(r) += 1.0;
                    }
                    row_factor_sum_orig(c) += v;
                }
            }

            // Pass 2: Obs accumulation parallelized across specificity columns.
            if (threads_use_obs > 1 && k > 1) {
                #pragma omp parallel for schedule(static) num_threads(threads_use_obs)
                for (arma::sword js = 0; js < static_cast<arma::sword>(k); ++js) {
                    const arma::uword j = static_cast<arma::uword>(js);
                    const double* h_col = H_norm_t.colptr(j);
                    double* obs_col = obs_orig.colptr(j);
                    for (arma::uword c = col_start; c < col_end; ++c) {
                        const unsigned long long p0 = op.indptr_[c]     - nnz_start;
                        const unsigned long long p1 = op.indptr_[c + 1] - nnz_start;
                        double acc = 0.0;
                        for (unsigned long long p = p0; p < p1; ++p) {
                            const arma::uword r = static_cast<arma::uword>(
                                (*indices)[static_cast<size_t>(p)]);
                            acc += (*data)[static_cast<size_t>(p)] * h_col[r];
                        }
                        obs_col[c] += acc;
                    }
                }
            } else {
                for (arma::uword c = col_start; c < col_end; ++c) {
                    const unsigned long long p0 = op.indptr_[c]     - nnz_start;
                    const unsigned long long p1 = op.indptr_[c + 1] - nnz_start;

                    for (unsigned long long p = p0; p < p1; ++p) {
                        const arma::uword r = static_cast<arma::uword>(
                            (*indices)[static_cast<size_t>(p)]);
                        const double v = (*data)[static_cast<size_t>(p)];
                        double* obs_base = obs_orig.memptr() + c;
                        for (arma::uword j = 0; j < k; ++j) {
                            obs_base[j * n_var] += v * H_norm_t(r, j);
                        }
                    }
                }
            }
        }
    }

    void backed_specificity_support_csr_(
        const BackedSparseMatrixOperator& op,
        const arma::mat& H_norm_t,
        arma::mat& obs_out,
        double shift,
        int thread_no)
    {
        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;
        const arma::uword k     = H_norm_t.n_cols;
        const arma::uword cs    = op.chunk_size_;
        const unsigned int threads_use_obs = actionet::get_num_threads(
            static_cast<unsigned int>(k), static_cast<unsigned int>(std::max(thread_no, 0)));

        if (shift == 0.0) return;

        const std::vector<double>* data;
        const std::vector<unsigned long long>* indices;

        for (arma::uword row_start = 0; row_start < n_obs; row_start += cs) {
            const arma::uword row_end = std::min<arma::uword>(n_obs, row_start + cs);
            const unsigned long long nnz_start = op.indptr_[row_start];
            const unsigned long long nnz_end   = op.indptr_[row_end];
            const unsigned long long nnz_count  = nnz_end - nnz_start;

            op.load_chunk_cached_(nnz_start, nnz_count, data, indices);

            if (threads_use_obs > 1 && k > 1) {
                #pragma omp parallel for schedule(static) num_threads(threads_use_obs)
                for (arma::sword js = 0; js < static_cast<arma::sword>(k); ++js) {
                    const arma::uword j = static_cast<arma::uword>(js);
                    const double* h_col = H_norm_t.colptr(j);
                    double* obs_col = obs_out.colptr(j);
                    for (arma::uword r = row_start; r < row_end; ++r) {
                        const double scaled_h = shift * h_col[r];
                        if (scaled_h == 0.0) continue;
                        const unsigned long long p0 = op.indptr_[r]     - nnz_start;
                        const unsigned long long p1 = op.indptr_[r + 1] - nnz_start;
                        for (unsigned long long p = p0; p < p1; ++p) {
                            const arma::uword c = static_cast<arma::uword>(
                                (*indices)[static_cast<size_t>(p)]);
                            obs_col[c] += scaled_h;
                        }
                    }
                }
            } else {
                for (arma::uword r = row_start; r < row_end; ++r) {
                    const unsigned long long p0 = op.indptr_[r]     - nnz_start;
                    const unsigned long long p1 = op.indptr_[r + 1] - nnz_start;

                    for (unsigned long long p = p0; p < p1; ++p) {
                        const arma::uword c = static_cast<arma::uword>(
                            (*indices)[static_cast<size_t>(p)]);
                        double* obs_base = obs_out.memptr() + c;
                        for (arma::uword j = 0; j < k; ++j) {
                            obs_base[j * n_var] += shift * H_norm_t(r, j);
                        }
                    }
                }
            }
        }
    }

    void backed_specificity_support_csc_(
        const BackedSparseMatrixOperator& op,
        const arma::mat& H_norm_t,
        arma::mat& obs_out,
        double shift,
        int thread_no)
    {
        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;
        const arma::uword k     = H_norm_t.n_cols;
        const arma::uword cs    = op.chunk_size_;
        const unsigned int threads_use_obs = actionet::get_num_threads(
            static_cast<unsigned int>(k), static_cast<unsigned int>(std::max(thread_no, 0)));

        if (shift == 0.0) return;

        const std::vector<double>* data;
        const std::vector<unsigned long long>* indices;

        for (arma::uword col_start = 0; col_start < n_var; col_start += cs) {
            const arma::uword col_end = std::min<arma::uword>(n_var, col_start + cs);
            const unsigned long long nnz_start = op.indptr_[col_start];
            const unsigned long long nnz_end   = op.indptr_[col_end];
            const unsigned long long nnz_count  = nnz_end - nnz_start;

            op.load_chunk_cached_(nnz_start, nnz_count, data, indices);

            if (threads_use_obs > 1 && k > 1) {
                #pragma omp parallel for schedule(static) num_threads(threads_use_obs)
                for (arma::sword js = 0; js < static_cast<arma::sword>(k); ++js) {
                    const arma::uword j = static_cast<arma::uword>(js);
                    const double* h_col = H_norm_t.colptr(j);
                    double* obs_col = obs_out.colptr(j);

                    for (arma::uword c = col_start; c < col_end; ++c) {
                        const unsigned long long p0 = op.indptr_[c]     - nnz_start;
                        const unsigned long long p1 = op.indptr_[c + 1] - nnz_start;
                        double acc = 0.0;
                        for (unsigned long long p = p0; p < p1; ++p) {
                            const arma::uword r = static_cast<arma::uword>(
                                (*indices)[static_cast<size_t>(p)]);
                            acc += h_col[r];
                        }
                        obs_col[c] += shift * acc;
                    }
                }
            } else {
                for (arma::uword c = col_start; c < col_end; ++c) {
                    const unsigned long long p0 = op.indptr_[c]     - nnz_start;
                    const unsigned long long p1 = op.indptr_[c + 1] - nnz_start;

                    for (unsigned long long p = p0; p < p1; ++p) {
                        const arma::uword r = static_cast<arma::uword>(
                            (*indices)[static_cast<size_t>(p)]);
                        double* obs_base = obs_out.memptr() + c;
                        for (arma::uword j = 0; j < k; ++j) {
                            obs_base[j * n_var] += shift * H_norm_t(r, j);
                        }
                    }
                }
            }
        }
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::mat& H, int thread_no) {
        stdout_printf("Computing feature specificity (backed sparse) ... ");

        const arma::uword n_obs = op.rows();
        const arma::uword n_var = op.cols();

        arma::mat H_norm = normalise_H(H);

        arma::vec row_count, col_count, row_factor_sum_orig;
        arma::mat obs_orig;
        double min_stored;

        if (op.isCSR()) {
            backed_specificity_scan_csr_(op, H_norm, row_count, col_count,
                                         row_factor_sum_orig, obs_orig, min_stored, thread_no);
        } else {
            backed_specificity_scan_csc_(op, H_norm, row_count, col_count,
                                         row_factor_sum_orig, obs_orig, min_stored, thread_no);
        }

        const double shift = (min_stored < 0.0) ? -min_stored : 0.0;
        arma::vec row_factor_sum = row_factor_sum_orig + shift * row_count;
        arma::mat Obs = obs_orig;

        if (shift > 0.0) {
            if (op.isCSR()) {
                backed_specificity_support_csr_(op, H_norm, Obs, shift, thread_no);
            } else {
                backed_specificity_support_csc_(op, H_norm, Obs, shift, thread_no);
            }
        }

        arma::vec row_factor = arma::zeros(n_var);
        for (arma::uword i = 0; i < n_var; ++i) {
            if (row_count(i) > 0.0) row_factor(i) = row_factor_sum(i) / row_count(i);
        }
        arma::vec row_p = row_count / static_cast<double>(n_obs);
        arma::vec col_p = col_count / static_cast<double>(n_var);

        stdout_printf("done\n");
        FLUSH;
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, n_obs, thread_no);
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no) {
        const arma::uword max_label = arma::max(labels);
        const arma::uword n_obs = op.rows();
        arma::mat H(n_obs, max_label, arma::fill::zeros);
        for (arma::uword j = 0; j < n_obs; ++j) {
            if (labels(j) > 0) H(j, labels(j) - 1) = 1.0;
        }
        return computeFeatureSpecificity(op, H, thread_no);
    }

} // namespace actionet

// ============================================================================
// Backed dense overloads
// ============================================================================

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
        row_count.zeros(n_var);
        col_count.zeros(n_obs);

        arma::mat slab;
        for (arma::uword obs_start = 0; obs_start < n_obs; obs_start += cs) {
            const arma::uword obs_count = std::min(cs, n_obs - obs_start);
            op.readSlab(obs_start, obs_count, slab);
            const double chunk_min = slab.min();
            if (chunk_min < global_min) global_min = chunk_min;

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

        if (!std::isfinite(global_min)) global_min = 0.0;
        if (global_min == 0.0) {
            return;
        }

        // Non-zero shift: recompute shifted row counts/sums exactly.
        row_factor_sum.zeros();
        row_count.zeros();
        col_count.zeros();

        for (arma::uword obs_start = 0; obs_start < n_obs; obs_start += cs) {
            const arma::uword obs_count = std::min(cs, n_obs - obs_start);
            op.readSlab(obs_start, obs_count, slab);
            slab -= global_min;

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
        return bernstein_tail_bounds(Obs, row_p, row_factor, col_p, H_norm, n_obs, thread_no);
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedDenseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no) {
        const arma::uword max_label = arma::max(labels);
        arma::mat H(op.rows(), max_label, arma::fill::zeros);
        for (arma::uword j = 0; j < op.rows(); ++j) {
            if (labels(j) > 0) H(j, labels(j) - 1) = 1.0;
        }
        return computeFeatureSpecificity(op, H, thread_no);
    }

} // namespace actionet
