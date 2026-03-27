#include "annotation/specificity.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "utils_internal/utils_matrix.hpp"

arma::field<arma::mat> getProbsObs(const arma::mat& S, const arma::mat& Ht, int thread_no) {
    arma::mat Sb = S;
    arma::uvec nnz_idx = arma::find(Sb > 0);
    (Sb(nnz_idx)).ones();

    // S is cells x genes.  gene density = sum over cell axis (dim=0).  cell density = sum over gene axis (dim=1).
    arma::vec row_p   = arma::trans(arma::sum(Sb, 0));      // gene-length  (was: sum over cols of genes x cells)
    arma::vec col_p   = arma::vec(arma::sum(Sb, 1));        // cell-length  (was: sum over rows of genes x cells)
    arma::vec row_factor = arma::trans(arma::sum(S, 0));    // gene-length row sums (pre-binarization)

    arma::field<arma::mat> out(4);
    out(0) = row_factor / row_p;       // mean of nonzero elements per gene
    out(1) = row_p / S.n_rows;         // gene density (n_rows = cells)
    out(2) = col_p / S.n_cols;         // cell density (n_cols = genes)
    out(3) = S.t() * Ht;              // (cells x genes)'(cells x k) = genes x k

    return (out);
}

arma::field<arma::mat> getProbsObs(const arma::sp_mat& S, const arma::mat& Ht, int thread_no) {
    // S is cells x genes.  row_p = gene density (n_genes length), col_p = cell density (n_cells length).
    // Only count elements that are strictly > 0 after the min-shift, matching the
    // dense path's `find(Sb > 0)` semantics.  Structural zeros and entries that
    // became exactly 0 after shift are excluded from density counts.
    arma::vec row_p = arma::zeros(S.n_cols);      // gene-length
    arma::vec col_p = arma::zeros(S.n_rows);      // cell-length
    arma::vec row_factor = arma::zeros(S.n_cols); // gene-length sum of values

    arma::sp_mat::const_iterator it = S.begin();
    arma::sp_mat::const_iterator it_end = S.end();
    for (; it != it_end; ++it) {
        const double v = (*it);
        if (v > 0.0) {
            col_p[it.row()]++;
            row_p[it.col()]++;
        }
        row_factor[it.col()] += v;
    }

    arma::field<arma::mat> out(4);
    out(0) = row_factor / row_p;       // mean of nonzero elements per gene
    out(1) = row_p / S.n_rows;         // gene density (n_rows = cells)
    out(2) = col_p / S.n_cols;         // cell density (n_cols = genes)
    out(3) = spmat_mat_product_parallel(S.t(), Ht, thread_no);  // (cells x genes)'(cells x k) = genes x k

    return (out);
}

namespace actionet {
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(const T& S, const arma::mat& H, int thread_no) {
        stdout_printf("Computing feature specificity ... ");

        // Non-mutating: compute the global minimum and work with a shifted copy.
        // S itself is never modified so callers retain the original values.
        double min_val = S.min();
        T S_shifted = S;
        if (min_val != 0.0) {
            S_shifted.for_each([min_val](typename T::elem_type& val) { val -= min_val; });
        }

        // H is cells x k (Plan 02).  Normalise each column (archetype) by its mean.
        arma::mat Ht = H;   // cells x k — passed directly to getProbsObs
        Ht.each_col([](arma::vec& h) {
            double mu = arma::mean(h);
            h /= (mu == 0) ? 1 : mu;
        }); // For numerical stability

        arma::field<arma::mat> p = getProbsObs(S_shifted, Ht, thread_no);

        arma::vec row_factor = p(0);  // genes-length: mean of nonzero per gene
        arma::vec row_p = p(1);       // genes-length: gene density
        arma::vec col_p = p(2);       // cells-length: cell density
        arma::mat Obs = p(3);         // genes x k

        double rho = arma::mean(col_p);
        arma::vec beta = col_p / rho; // Relative density compared to the overall density
        arma::mat Gamma = Ht;         // cells x k
        arma::vec a(H.n_cols);        // k-length (H.n_cols = k)
        for (int i = 0; i < (int)H.n_cols; i++) {
            Gamma.col(i) %= beta;
            a(i) = arma::max(Gamma.col(i));
        }

        arma::mat Exp = (row_p % row_factor) * arma::sum(Gamma, 0);           // genes x k
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
        res(0) = Obs / Ht.n_rows;     // average profile: genes x k (divide by n_cells)
        res(1) = logPvals_upper;
        res(2) = logPvals_lower;

        return (res);
    }

    template arma::field<arma::mat> computeFeatureSpecificity<arma::mat>(const arma::mat& S, const arma::mat& H, int thread_no);
    template arma::field<arma::mat> computeFeatureSpecificity<arma::sp_mat>(
        const arma::sp_mat& S, const arma::mat& H, int thread_no);

    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(const T& S, const arma::uvec& labels, int thread_no) {
        // S is cells x genes.  n_cells = S.n_rows.
        // H is built as cells x k (matching the new contract).
        arma::mat H(S.n_rows, arma::max(labels), arma::fill::zeros);  // cells x k

        for (int i = 1; i <= (int)arma::max(labels); i++) {
            arma::uvec idx = arma::find(labels == (arma::uword)i);
            for (arma::uword j : idx) {
                H(j, i - 1) = 1.0;
            }
        }

        arma::field<arma::mat> res = computeFeatureSpecificity(S, H, thread_no);

        return (res);
    }

    template arma::field<arma::mat> computeFeatureSpecificity<arma::mat>(
        const arma::mat& S, const arma::uvec& labels, int thread_no);
    template arma::field<arma::mat> computeFeatureSpecificity<arma::sp_mat>(
        const arma::sp_mat& S, const arma::uvec& labels, int thread_no);
} // namespace actionet

// ============================================================================
// Backed sparse overloads (non-template; friend access to
// BackedSparseMatrixOperator private members).
// ============================================================================

namespace actionet {

    /// Single-pass streaming specificity for a CSR-backed HDF5 matrix.
    ///
    /// Iterates over row chunks.  For each stored element (obs_row r, var_col c,
    /// logical value v = transform_value_(r, raw)):
    ///   - Tracks global stored minimum for the shift correction.
    ///   - Accumulates per-feature (col) and per-cell (row) nnz counts.
    ///   - Accumulates row_factor_sum_orig[c] += v  (pre-shift column sums).
    ///   - Accumulates obs_orig(c, :)    += v * H_norm(r, :)
    ///   - Accumulates support_obs(c, :) +=     H_norm(r, :)  (binary support)
    ///
    /// After the scan the min-shift is applied analytically:
    ///   row_factor_sum = row_factor_sum_orig + shift * row_count
    ///   Obs            = obs_orig            + shift * support_obs
    void backed_specificity_scan_csr_(
        const BackedSparseMatrixOperator& op,
        const arma::mat& H_norm_t,           // shape: n_obs x k  (i.e. Ht in C++ convention)
        arma::vec& row_count,                // out: n_var
        arma::vec& col_count,                // out: n_obs
        arma::vec& row_factor_sum_orig,      // out: n_var
        arma::mat& obs_orig,                 // out: n_var x k
        arma::mat& support_obs,              // out: n_var x k
        double& min_stored)                  // out: minimum stored value
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

                    if (v < min_stored) {
                        min_stored = v;
                    }
                    if (v > 0.0) {
                        row_count(c)           += 1.0;
                        col_count(r)           += 1.0;
                    }
                    row_factor_sum_orig(c) += v;
                    obs_orig.row(c)        += v * h_row;
                    support_obs.row(c)     += h_row;
                }
            }
        }
    }

    /// Single-pass streaming specificity for a CSC-backed HDF5 matrix.
    ///
    /// In CSC format the indptr covers var columns; each stored entry
    /// (var_col c = current col, obs_row r = indices[p]) maps to operator
    /// position S(c, r).  The accumulation is identical to the CSR case but
    /// the outer loop is over var chunks rather than obs chunks.
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
                    // In CSC, the stored row index is obs_row, and the column is var_col.
                    // transform_value_ requires obs_row as the first argument.
                    const double v = op.transform_value_(r, (*data)[static_cast<size_t>(p)]);

                    if (v < min_stored) {
                        min_stored = v;
                    }
                    if (v > 0.0) {
                        row_count(c)           += 1.0;
                        col_count(r)           += 1.0;
                    }
                    row_factor_sum_orig(c) += v;

                    const arma::rowvec h_row = H_norm_t.row(r);
                    obs_orig.row(c)        += v * h_row;
                    support_obs.row(c)     += h_row;
                }
            }
        }
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::mat& H, int /*thread_no*/) {
        stdout_printf("Computing feature specificity (backed sparse) ... ");

        const arma::uword n_obs = op.n_obs_;
        const arma::uword n_var = op.n_var_;
        const arma::uword k     = static_cast<arma::uword>(H.n_cols);  // H is cells x k (Plan 02)

        // Normalise H column-wise (matching the in-memory path's Ht treatment).
        // H is (n_obs x k) — each column corresponds to one group/archetype.
        arma::mat H_norm_t = H;  // n_obs x k — no transpose needed (was H.t() when H was k x n_obs)
        for (arma::uword j = 0; j < k; ++j) {
            const double mu = arma::mean(H_norm_t.col(j));
            if (mu != 0.0) {
                H_norm_t.col(j) /= mu;
            }
        }

        // Single-pass scan: accumulate all per-feature and per-cell statistics
        // together with the raw (pre-shift) Obs and support matrices.
        arma::vec row_count, col_count, row_factor_sum_orig;
        arma::mat obs_orig, support_obs;
        double min_stored;

        if (op.is_csr_) {
            backed_specificity_scan_csr_(op, H_norm_t,
                                         row_count, col_count,
                                         row_factor_sum_orig,
                                         obs_orig, support_obs,
                                         min_stored);
        } else {
            backed_specificity_scan_csc_(op, H_norm_t,
                                         row_count, col_count,
                                         row_factor_sum_orig,
                                         obs_orig, support_obs,
                                         min_stored);
        }

        // Apply the min-shift analytically so we never mutate the HDF5 data.
        // shift = -min(0, min_stored), i.e. only shift if there are negative values.
        const double shift = (min_stored < 0.0) ? -min_stored : 0.0;

        arma::vec row_factor_sum = row_factor_sum_orig + shift * row_count;
        arma::mat Obs = obs_orig + shift * support_obs;

        // Per-feature mean of non-zero elements (row_factor) and density (row_p).
        arma::vec row_factor = arma::zeros(n_var);
        for (arma::uword i = 0; i < n_var; ++i) {
            if (row_count(i) > 0.0) {
                row_factor(i) = row_factor_sum(i) / row_count(i);
            }
        }
        arma::vec row_p = row_count  / static_cast<double>(n_obs);
        arma::vec col_p = col_count  / static_cast<double>(n_var);

        const double rho = arma::mean(col_p);
        arma::vec beta = (rho == 0.0) ? arma::zeros(n_obs) : arma::vec(col_p / rho);

        // Gamma = H_norm_t scaled by beta; shape (n_obs x k).
        arma::mat Gamma = H_norm_t;
        arma::vec a(k);
        for (arma::uword j = 0; j < k; ++j) {
            Gamma.col(j) %= beta;
            a(j) = arma::max(Gamma.col(j));
        }

        // Bernstein tail bounds (identical math to in-memory path).
        arma::mat Exp    = (row_p % row_factor) * arma::sum(Gamma, 0);
        arma::mat Nu     = (row_p % arma::square(row_factor)) * arma::sum(arma::square(Gamma), 0);
        arma::mat A      = row_factor * arma::trans(a);
        arma::mat Lambda = Obs - Exp;

        arma::mat logPvals_lower = arma::square(Lambda) / (2.0 * Nu);
        arma::uvec uidx = arma::find(Lambda >= 0);
        logPvals_lower(uidx).zeros();
        logPvals_lower.replace(arma::datum::nan, 0.0);

        arma::mat logPvals_upper = arma::square(Lambda) / (2.0 * (Nu + (Lambda % A / 3.0)));
        arma::uvec lidx = arma::find(Lambda <= 0);
        logPvals_upper(lidx).zeros();
        logPvals_upper.replace(arma::datum::nan, 0.0);

        const double log10_e = std::log(10.0);
        logPvals_lower /= log10_e;
        logPvals_upper /= log10_e;

        stdout_printf("done\n");
        FLUSH;

        arma::field<arma::mat> res(3);
        res(0) = Obs / static_cast<double>(n_obs);
        res(1) = logPvals_upper;
        res(2) = logPvals_lower;
        return res;
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no) {
        const arma::uword max_label = arma::max(labels);
        const arma::uword n_obs     = op.n_obs_;

        // H is cells x k (Plan 02)
        arma::mat H(n_obs, max_label, arma::fill::zeros);
        for (arma::uword i = 1; i <= max_label; ++i) {
            arma::uvec idx = arma::find(labels == i);
            for (arma::uword j : idx) {
                H(j, i - 1) = 1.0;
            }
        }

        return computeFeatureSpecificity(op, H, thread_no);
    }

} // namespace actionet

// ============================================================================
// Backed dense overloads (use BackedDenseMatrixOperator for HDF5 dense datasets).
// ============================================================================
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

namespace actionet {

    /// Single-pass scan over dense HDF5 obs-chunks to find the global minimum
    /// and accumulate per-feature (column) sums and nnz-after-shift counts.
    ///
    /// Matches the in-memory ``getProbsObs(arma::mat)`` behaviour:
    ///   - shift = global_min (unconditional; ensures min(S_shifted) == 0)
    ///   - row_count(c) = number of elements in column c that are > 0 after shift
    ///     (i.e. all except the single global-minimum element per column)
    ///   - col_count(r) = number of elements in row r that are > 0 after shift
    ///   - row_factor_sum(c) = sum of all elements in column c of S_shifted
    ///
    /// Unlike the sparse-backed path (which initialises min_stored = 0 and only
    /// records the minimum of the stored NNZ entries) the dense path always applies
    /// a full shift so that Bernstein density estimates match the in-memory path
    /// exactly.
    static void backed_dense_specificity_scan_(
        BackedDenseMatrixOperator& op,
        arma::vec& row_count,        // out: nnz > 0 count per gene (n_var)
        arma::vec& col_count,        // out: nnz > 0 count per obs  (n_obs)
        arma::vec& row_factor_sum,   // out: column sums of S_shifted
        double&    global_min)
    {
        const arma::uword n_obs = op.rows();
        const arma::uword n_var = op.cols();
        const arma::uword cs    = op.effectiveChunkSize();

        row_factor_sum.zeros(n_var);
        global_min = std::numeric_limits<double>::infinity();

        // --- Pass 1: find global minimum ---
        arma::mat slab;
        for (arma::uword obs_start = 0; obs_start < n_obs; obs_start += cs) {
            const arma::uword obs_count = std::min(cs, n_obs - obs_start);
            op.readSlab(obs_start, obs_count, slab);
            const double chunk_min = slab.min();
            if (chunk_min < global_min) global_min = chunk_min;
        }
        if (!std::isfinite(global_min)) global_min = 0.0;

        // --- Pass 2: accumulate column sums and nnz-after-shift counts ---
        row_count.zeros(n_var);
        col_count.zeros(n_obs);

        for (arma::uword obs_start = 0; obs_start < n_obs; obs_start += cs) {
            const arma::uword obs_count = std::min(cs, n_obs - obs_start);
            op.readSlab(obs_start, obs_count, slab);

            // Apply shift: S_chunk_shifted = slab - global_min
            if (global_min != 0.0) {
                slab -= global_min;
            }

            // Column sums of S_shifted
            row_factor_sum += arma::sum(slab, 0).t();

            // Count elements > 0 per gene (column) and per obs (row)
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
        const arma::uword k     = static_cast<arma::uword>(H.n_cols);

        // Normalise H column-wise (matching the in-memory and sparse-backed paths).
        arma::mat H_norm = H;  // n_obs x k
        for (arma::uword j = 0; j < k; ++j) {
            const double mu = arma::mean(H_norm.col(j));
            if (mu != 0.0) {
                H_norm.col(j) /= mu;
            }
        }

        // --- Accumulation pass: global min (2 passes: find min, then accumulate on shifted) ---
        arma::vec row_count(n_var), col_count(n_obs), row_factor_sum(n_var);
        double global_min;
        backed_dense_specificity_scan_(op, row_count, col_count, row_factor_sum, global_min);

        // row_factor_sum already corresponds to the shifted S (scan applied shift internally).
        // Shift for Obs correction: S_shifted = S - global_min  =>  Obs_shifted = Obs - global_min * colsums(H_norm)
        // Use rmatmat to get Obs = S.t() * H_norm  (unshifted), then subtract the correction.
        arma::mat Obs;
        op.rmatmat(H_norm, Obs);  // Obs = S.t() * H_norm  (n_var x k), unshifted

        if (global_min != 0.0) {
            // S_shifted = S - global_min  =>  S_shifted.t()*H = S.t()*H - global_min * sum(H, 0)
            arma::rowvec H_norm_colsums = arma::sum(H_norm, 0);  // (1 x k)
            Obs -= global_min * arma::ones(n_var, 1) * H_norm_colsums;
        }

        // --- Tail-bound math (identical to sparse-backed path) ---
        arma::vec row_factor = arma::zeros(n_var);
        for (arma::uword i = 0; i < n_var; ++i) {
            if (row_count(i) > 0.0) {
                row_factor(i) = row_factor_sum(i) / row_count(i);
            }
        }
        arma::vec row_p = row_count  / static_cast<double>(n_obs);
        arma::vec col_p = col_count  / static_cast<double>(n_var);

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
        arma::uvec uidx = arma::find(Lambda >= 0);
        logPvals_lower(uidx).zeros();
        logPvals_lower.replace(arma::datum::nan, 0.0);

        arma::mat logPvals_upper = arma::square(Lambda) / (2.0 * (Nu + (Lambda % A / 3.0)));
        arma::uvec lidx = arma::find(Lambda <= 0);
        logPvals_upper(lidx).zeros();
        logPvals_upper.replace(arma::datum::nan, 0.0);

        const double log10_e = std::log(10.0);
        logPvals_lower /= log10_e;
        logPvals_upper /= log10_e;

        stdout_printf("done\n");
        FLUSH;

        arma::field<arma::mat> res(3);
        res(0) = Obs / static_cast<double>(n_obs);
        res(1) = logPvals_upper;
        res(2) = logPvals_lower;
        return res;
    }

    arma::field<arma::mat> computeFeatureSpecificity(BackedDenseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no) {
        const arma::uword max_label = arma::max(labels);
        const arma::uword n_obs     = op.rows();

        arma::mat H(n_obs, max_label, arma::fill::zeros);
        for (arma::uword i = 1; i <= max_label; ++i) {
            arma::uvec idx = arma::find(labels == i);
            for (arma::uword j : idx) {
                H(j, i - 1) = 1.0;
            }
        }

        return computeFeatureSpecificity(op, H, thread_no);
    }

} // namespace actionet
