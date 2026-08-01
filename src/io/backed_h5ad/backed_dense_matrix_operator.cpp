#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"
#include "io/backed_h5ad/h5ad_matrix_io.hpp"

#include "_h5_utils.hpp"

#include "fastapprox/fastlog.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
    using actionet::detail::h5::check_h5;
} // namespace

namespace actionet {

    BackedDenseMatrixOperator::BackedDenseMatrixOperator(
        const std::string& file_path,
        const std::string& group_path,
        arma::uword chunk_size,
        const std::vector<double>& row_scale_factors,
        bool apply_log1p,
        double log_scale,
        size_t slab_byte_budget,
        int n_threads)
        : file_path_(file_path),
          group_path_(group_path),
          apply_log1p_(apply_log1p),
          log_scale_(log_scale),
          chunk_size_(std::max<arma::uword>(1, chunk_size)),
          effective_chunk_size_(0),
          n_obs_(0),
          n_var_(0),
          n_threads_(static_cast<unsigned int>(std::max(0, n_threads))),
          file_id_(-1),
          dataset_id_(-1) {

        const auto matrix_info = h5ad::inspect_matrix(file_path_, group_path_);
        check_h5(
            matrix_info.encoding == h5ad::MatrixEncoding::Dense,
            "BackedDenseMatrixOperator requires a dense H5AD matrix");
        check_h5(
            matrix_info.rows <= std::numeric_limits<arma::uword>::max() &&
                matrix_info.cols <= std::numeric_limits<arma::uword>::max(),
            "Dense H5AD shape exceeds the compute operator index range");
        n_obs_ = static_cast<arma::uword>(matrix_info.rows);
        n_var_ = static_cast<arma::uword>(matrix_info.cols);

        // A throw during construction does NOT run the destructor, so guard the
        // open/validate sequence and release partially-acquired handles before
        // rethrowing to avoid leaking the file/dataset ids.
        try {
        file_id_ = actionet::detail::h5::open_h5_readonly_no_lock(
            file_path_, "BackedDenseMatrixOperator");

        dataset_id_ = H5Dopen2(file_id_, group_path_.c_str(), H5P_DEFAULT);
        check_h5(dataset_id_ >= 0,
                 "Failed to open dense dataset path (expected a 2D dataset, "
                 "not a sparse group)");

        // Clamp effective chunk size to the byte budget.
        // Each slab row is n_var_ doubles (8 bytes each).
        arma::uword max_rows_by_budget = 1;
        if (n_var_ > 0) {
            size_t bytes_per_row = static_cast<size_t>(n_var_) * sizeof(double);
            max_rows_by_budget = static_cast<arma::uword>(
                std::max<size_t>(1, slab_byte_budget / bytes_per_row));
        }
        effective_chunk_size_ = std::min(chunk_size_, max_rows_by_budget);

        if (!row_scale_factors.empty()) {
            check_h5(row_scale_factors.size() == static_cast<size_t>(n_obs_),
                     "row_scale_factors length must equal n_obs");
            row_scale_ = arma::vec(row_scale_factors);
        }
        check_h5(std::isfinite(log_scale_) && log_scale_ > 0.0,
                 "log_scale must be finite and > 0");
        } catch (...) {
            close_handles_();
            throw;
        }
    }

    void BackedDenseMatrixOperator::close_handles_() {
        if (dataset_id_ >= 0) { H5Dclose(dataset_id_); dataset_id_ = -1; }
        if (file_id_ >= 0) { H5Fclose(file_id_); file_id_ = -1; }
    }

    BackedDenseMatrixOperator::~BackedDenseMatrixOperator() {
        close_handles_();
    }

    BackedDenseMatrixOperator::BackedDenseMatrixOperator(BackedDenseMatrixOperator&& other) noexcept
        : file_path_(std::move(other.file_path_)),
          group_path_(std::move(other.group_path_)),
          apply_log1p_(other.apply_log1p_),
          log_scale_(other.log_scale_),
          chunk_size_(other.chunk_size_),
          effective_chunk_size_(other.effective_chunk_size_),
          n_obs_(other.n_obs_),
          n_var_(other.n_var_),
          row_scale_(std::move(other.row_scale_)),
          n_threads_(other.n_threads_),
          file_id_(other.file_id_),
          dataset_id_(other.dataset_id_) {
        other.file_id_ = -1;
        other.dataset_id_ = -1;
    }

    BackedDenseMatrixOperator& BackedDenseMatrixOperator::operator=(BackedDenseMatrixOperator&& other) noexcept {
        if (this != &other) {
            close_handles_();
            file_path_ = std::move(other.file_path_);
            group_path_ = std::move(other.group_path_);
            apply_log1p_ = other.apply_log1p_;
            log_scale_ = other.log_scale_;
            chunk_size_ = other.chunk_size_;
            effective_chunk_size_ = other.effective_chunk_size_;
            n_obs_ = other.n_obs_;
            n_var_ = other.n_var_;
            row_scale_ = std::move(other.row_scale_);
            n_threads_ = other.n_threads_;
            file_id_ = other.file_id_;
            dataset_id_ = other.dataset_id_;
            other.file_id_ = -1;
            other.dataset_id_ = -1;
        }
        return *this;
    }

    void BackedDenseMatrixOperator::read_slab_(
        arma::uword obs_start, arma::uword obs_count, arma::mat& slab) const {

        slab.set_size(obs_count, n_var_);
        if (obs_count == 0) return;

        actionet::detail::h5::Space file_space(H5Dget_space(dataset_id_));
        check_h5(static_cast<bool>(file_space), "Failed to get dense dataset dataspace");

        hsize_t offset[2] = {static_cast<hsize_t>(obs_start), 0};
        hsize_t count[2] = {static_cast<hsize_t>(obs_count), static_cast<hsize_t>(n_var_)};
        check_h5(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, offset, nullptr, count, nullptr) >= 0,
                 "Failed to select dense hyperslab");

        actionet::detail::h5::Space mem_space(H5Screate_simple(2, count, nullptr));
        check_h5(static_cast<bool>(mem_space), "Failed to create dense memory dataspace");

        // HDF5 writes the (obs_count × n_var) hyperslab in row-major order.
        // Read it into an arma::mat that is dimensioned as (n_var × obs_count):
        // arma is column-major, so treating the row-major buffer as a
        // (n_var × obs_count) column-major matrix reinterprets each on-disk
        // row as one column.  A single ``strans`` then yields the desired
        // (obs_count × n_var) slab without an explicit per-element copy.
        arma::mat tmp(n_var_, obs_count);
        check_h5(H5Dread(dataset_id_, H5T_NATIVE_DOUBLE, mem_space.get(), file_space.get(),
                         H5P_DEFAULT, tmp.memptr()) >= 0,
                 "Failed to read dense slab");
        slab = tmp.t();
    }

    void BackedDenseMatrixOperator::apply_transforms_(
        arma::uword obs_start, arma::mat& slab) const {

        const arma::uword nrows = slab.n_rows;
        const arma::uword ncols = slab.n_cols;
        const bool has_scale = !row_scale_.is_empty();
        const bool apply_log_scale = apply_log1p_ && std::abs(log_scale_ - 1.0) > 0.0;

        if (!has_scale && !apply_log1p_) return;

        // Row scaling: vectorised via arma's row-wise scalar multiplication.
        // slab is (obs_count × n_var); row_scale_ is (n_obs,).  Using
        // ``.row(r) *= factor`` in a tight loop lets arma dispatch to BLAS
        // scal on each row without the C-level per-element multiply.
        if (has_scale) {
            for (arma::uword r = 0; r < nrows; ++r) {
                slab.row(r) *= row_scale_(obs_start + r);
            }
        }

        // Log1p: element-wise operation on a contiguous column-major buffer.
        // Parallelise over columns (the outer arma dimension) so each thread
        // writes to a disjoint contiguous span of memory.
        if (apply_log1p_) {
            const unsigned int threads_use = actionet::get_num_threads_nested_safe(
                static_cast<unsigned int>(ncols), n_threads_);
            #pragma omp parallel for schedule(static) num_threads(threads_use) if(threads_use > 1 && ncols > 1)
            for (arma::sword cs = 0; cs < static_cast<arma::sword>(ncols); ++cs) {
                double* col = slab.colptr(static_cast<arma::uword>(cs));
                for (arma::uword r = 0; r < nrows; ++r) {
                    double val = static_cast<double>(fastlog(1.0f + static_cast<float>(col[r])));
                    if (apply_log_scale) val *= log_scale_;
                    col[r] = val;
                }
            }
        }
    }

    // S = X (obs x var), operator shape: rows = n_obs_, cols = n_var_.
    // matvec: y = S * x, where x is (n_var,), y is (n_obs,) — S is cells x genes.
    // Iterate over obs-chunks of X, each chunk is (obs_count x n_var); y_chunk = chunk * x.
    void BackedDenseMatrixOperator::matvec(const arma::vec& x, arma::vec& y) const {
        if (x.n_elem != n_var_) {
            throw std::runtime_error("BackedDenseMatrixOperator::matvec dimension mismatch");
        }

        y.set_size(n_obs_);
        arma::mat slab;

        for (arma::uword obs_start = 0; obs_start < n_obs_; obs_start += effective_chunk_size_) {
            const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
            const arma::uword obs_count = obs_end - obs_start;

            read_slab_(obs_start, obs_count, slab);
            apply_transforms_(obs_start, slab);

            // y_chunk = slab * x  =>  (obs_count x n_var) * (n_var,) = (obs_count,)
            y.subvec(obs_start, obs_end - 1) = slab * x;
        }
    }

    // rmatvec: y = S' * x, where x is (n_obs,), y is (n_var,).
    // Iterate over obs-chunks; accumulate slab.t() * x_chunk into y.
    void BackedDenseMatrixOperator::rmatvec(const arma::vec& x, arma::vec& y) const {
        if (x.n_elem != n_obs_) {
            throw std::runtime_error("BackedDenseMatrixOperator::rmatvec dimension mismatch");
        }

        y.zeros(n_var_);
        arma::mat slab;

        for (arma::uword obs_start = 0; obs_start < n_obs_; obs_start += effective_chunk_size_) {
            const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
            const arma::uword obs_count = obs_end - obs_start;

            read_slab_(obs_start, obs_count, slab);
            apply_transforms_(obs_start, slab);

            // y += slab.t() * x_chunk  =>  (n_var x obs_count) * (obs_count,) = (n_var,)
            y += slab.t() * x.subvec(obs_start, obs_end - 1);
        }
    }

    // matmat: Y = S * X, where X is (n_var, k), Y is (n_obs, k).
    void BackedDenseMatrixOperator::matmat(const arma::mat& X, arma::mat& Y) const {
        if (X.n_rows != n_var_) {
            throw std::runtime_error("BackedDenseMatrixOperator::matmat dimension mismatch");
        }

        Y.set_size(n_obs_, X.n_cols);
        arma::mat slab;

        for (arma::uword obs_start = 0; obs_start < n_obs_; obs_start += effective_chunk_size_) {
            const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
            const arma::uword obs_count = obs_end - obs_start;

            read_slab_(obs_start, obs_count, slab);
            apply_transforms_(obs_start, slab);

            // Y_chunk = slab * X  =>  (obs_count x n_var) * (n_var x k) = (obs_count x k)
            Y.rows(obs_start, obs_end - 1) = slab * X;
        }
    }

    // rmatmat: Y = S' * X, where X is (n_obs, k), Y is (n_var, k).
    void BackedDenseMatrixOperator::rmatmat(const arma::mat& X, arma::mat& Y) const {
        if (X.n_rows != n_obs_) {
            throw std::runtime_error("BackedDenseMatrixOperator::rmatmat dimension mismatch");
        }

        Y.zeros(n_var_, X.n_cols);
        arma::mat slab;

        for (arma::uword obs_start = 0; obs_start < n_obs_; obs_start += effective_chunk_size_) {
            const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
            const arma::uword obs_count = obs_end - obs_start;

            read_slab_(obs_start, obs_count, slab);
            apply_transforms_(obs_start, slab);

            // Y += slab.t() * X_chunk  =>  (n_var x obs_count) * (obs_count x k) = (n_var x k)
            Y += slab.t() * X.rows(obs_start, obs_end - 1);
        }
    }

    // ---- rowStats -----------------------------------------------------

    void BackedDenseMatrixOperator::rowStats(arma::vec& row_sum,
                                             arma::vec& row_sum_sq,
                                             arma::vec& nnz) const {
        row_sum.zeros(n_obs_);
        row_sum_sq.zeros(n_obs_);
        nnz.zeros(n_obs_);

        arma::mat slab;
        for (arma::uword obs_start = 0; obs_start < n_obs_; obs_start += effective_chunk_size_) {
            const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
            const arma::uword obs_count = obs_end - obs_start;
            read_slab_(obs_start, obs_count, slab);
            apply_transforms_(obs_start, slab);

            for (arma::uword r = 0; r < obs_count; ++r) {
                double s = 0.0, ss = 0.0;
                double c_nnz = 0.0;
                for (arma::uword c = 0; c < n_var_; ++c) {
                    const double v = slab(r, c);
                    if (v != 0.0) {
                        s += v;
                        ss += v * v;
                        c_nnz += 1.0;
                    }
                }
                row_sum(obs_start + r)    = s;
                row_sum_sq(obs_start + r) = ss;
                nnz(obs_start + r)        = c_nnz;
            }
        }
    }

    // ---- takeColumns implementations ------------------------------------------------

    arma::mat BackedDenseMatrixOperator::takeColumnsDense(
        const arma::uvec& col_indices,
        const arma::uvec& row_indices) const {

        for (arma::uword i = 0; i < col_indices.n_elem; ++i) {
            if (col_indices(i) >= n_var_) {
                throw std::out_of_range(
                    "BackedDenseMatrixOperator::takeColumnsDense column index out of range");
            }
        }
        for (arma::uword i = 0; i < row_indices.n_elem; ++i) {
            if (row_indices(i) >= n_obs_) {
                throw std::out_of_range(
                    "BackedDenseMatrixOperator::takeColumnsDense row index out of range");
            }
        }

        const arma::uword n_sel = col_indices.n_elem;
        const bool subset_rows = !row_indices.is_empty();
        const arma::uword n_out_rows = subset_rows ? row_indices.n_elem : n_obs_;

        if (n_sel == 0) {
            return arma::mat(n_out_rows, 0);
        }

        // Direct gather over dense-slab chunks: read one obs-chunk at a
        // time, apply the lazy transform, then copy the requested columns
        // straight into the output.  Avoids the previous approach of
        // building a full (n_var × n_sel) sparse selector, densifying it
        // (n_var × n_sel doubles), and running matmat — which allocated
        // ~n_var × n_sel × 8 bytes of scratch and did an O(n_var·n_sel)
        // dense multiply for what is inherently an O(n_obs·n_sel) gather.
        arma::mat out(n_out_rows, n_sel);
        arma::mat slab;

        if (!subset_rows) {
            for (arma::uword obs_start = 0; obs_start < n_obs_; obs_start += effective_chunk_size_) {
                const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
                const arma::uword obs_count = obs_end - obs_start;

                read_slab_(obs_start, obs_count, slab);
                apply_transforms_(obs_start, slab);
                // ``slab.cols(col_indices)`` gathers the requested columns
                // (duplicates preserved) in a single armadillo call.
                out.rows(obs_start, obs_end - 1) = slab.cols(col_indices);
            }
            return out;
        }

        // Row-subset path: bucket requested rows into their originating
        // obs-chunks so each chunk is read at most once, then gather
        // (rows, cols) directly from the slab into the correct output row.
        std::vector<std::vector<arma::uword>> per_chunk_out_indices;
        std::vector<arma::uword> per_chunk_start;
        {
            const arma::uword n_req = row_indices.n_elem;
            std::vector<arma::uword> perm(n_req);
            for (arma::uword i = 0; i < n_req; ++i) perm[i] = i;
            std::sort(perm.begin(), perm.end(),
                      [&](arma::uword a, arma::uword b) {
                          return row_indices(a) < row_indices(b);
                      });

            arma::uword cursor = 0;
            for (arma::uword obs_start = 0; obs_start < n_obs_ && cursor < n_req;
                 obs_start += effective_chunk_size_) {
                const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
                std::vector<arma::uword> out_idx;
                out_idx.reserve(n_req);
                while (cursor < n_req && row_indices(perm[cursor]) < obs_end) {
                    out_idx.push_back(perm[cursor]);
                    ++cursor;
                }
                if (!out_idx.empty()) {
                    per_chunk_start.push_back(obs_start);
                    per_chunk_out_indices.push_back(std::move(out_idx));
                }
            }
        }

        for (size_t k = 0; k < per_chunk_start.size(); ++k) {
            const arma::uword obs_start = per_chunk_start[k];
            const arma::uword obs_end = std::min(n_obs_, obs_start + effective_chunk_size_);
            const arma::uword obs_count = obs_end - obs_start;

            read_slab_(obs_start, obs_count, slab);
            apply_transforms_(obs_start, slab);

            for (arma::uword out_i : per_chunk_out_indices[k]) {
                const arma::uword src_row = row_indices(out_i) - obs_start;
                for (arma::uword j = 0; j < n_sel; ++j) {
                    out(out_i, j) = slab(src_row, col_indices(j));
                }
            }
        }
        return out;
    }

    arma::sp_mat BackedDenseMatrixOperator::takeColumnsSparse(
        const arma::uvec& col_indices,
        const arma::uvec& row_indices) const {

        for (arma::uword i = 0; i < col_indices.n_elem; ++i) {
            if (col_indices(i) >= n_var_) {
                throw std::out_of_range(
                    "BackedDenseMatrixOperator::takeColumnsSparse column index out of range");
            }
        }
        for (arma::uword i = 0; i < row_indices.n_elem; ++i) {
            if (row_indices(i) >= n_obs_) {
                throw std::out_of_range(
                    "BackedDenseMatrixOperator::takeColumnsSparse row index out of range");
            }
        }

        arma::mat dense = takeColumnsDense(col_indices, row_indices);
        return arma::sp_mat(dense);
    }

} // namespace actionet
