#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
    void check_h5(bool ok, const char* msg) {
        if (!ok) {
            throw std::runtime_error(msg);
        }
    }
} // namespace

namespace actionet {

    std::vector<long long> BackedDenseMatrixOperator::read_shape_(hid_t dataset_id) {
        hid_t space_id = H5Dget_space(dataset_id);
        check_h5(space_id >= 0, "Failed to get dense dataset dataspace");

        int ndims = H5Sget_simple_extent_ndims(space_id);
        check_h5(ndims == 2, "Dense backed dataset must be 2D");

        hsize_t dims[2] = {0, 0};
        check_h5(H5Sget_simple_extent_dims(space_id, dims, nullptr) == 2,
                 "Failed to read dense dataset dimensions");
        H5Sclose(space_id);

        return {static_cast<long long>(dims[0]), static_cast<long long>(dims[1])};
    }

    BackedDenseMatrixOperator::BackedDenseMatrixOperator(
        const std::string& file_path,
        const std::string& group_path,
        arma::uword chunk_size,
        const std::vector<double>& row_scale_factors,
        bool apply_log1p,
        size_t slab_byte_budget)
        : file_path_(file_path),
          group_path_(group_path),
          apply_log1p_(apply_log1p),
          chunk_size_(std::max<arma::uword>(1, chunk_size)),
          effective_chunk_size_(0),
          n_obs_(0),
          n_var_(0),
          file_id_(-1),
          dataset_id_(-1) {

        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        check_h5(fapl >= 0, "Failed to create file access property list");
        H5Pset_file_locking(fapl, 0, 1);
        file_id_ = H5Fopen(file_path_.c_str(), H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        check_h5(file_id_ >= 0, "Failed to open h5ad file");

        dataset_id_ = H5Dopen2(file_id_, group_path_.c_str(), H5P_DEFAULT);
        check_h5(dataset_id_ >= 0,
                 "Failed to open dense dataset path (expected a 2D dataset, "
                 "not a sparse group)");

        auto shape = read_shape_(dataset_id_);
        check_h5(shape[0] >= 0 && shape[1] >= 0, "Dense shape must be non-negative");
        n_obs_ = static_cast<arma::uword>(shape[0]);
        n_var_ = static_cast<arma::uword>(shape[1]);

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
          chunk_size_(other.chunk_size_),
          effective_chunk_size_(other.effective_chunk_size_),
          n_obs_(other.n_obs_),
          n_var_(other.n_var_),
          row_scale_(std::move(other.row_scale_)),
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
            chunk_size_ = other.chunk_size_;
            effective_chunk_size_ = other.effective_chunk_size_;
            n_obs_ = other.n_obs_;
            n_var_ = other.n_var_;
            row_scale_ = std::move(other.row_scale_);
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

        hid_t file_space = H5Dget_space(dataset_id_);
        check_h5(file_space >= 0, "Failed to get dense dataset dataspace");

        hsize_t offset[2] = {static_cast<hsize_t>(obs_start), 0};
        hsize_t count[2] = {static_cast<hsize_t>(obs_count), static_cast<hsize_t>(n_var_)};
        check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, nullptr, count, nullptr) >= 0,
                 "Failed to select dense hyperslab");

        hid_t mem_space = H5Screate_simple(2, count, nullptr);
        check_h5(mem_space >= 0, "Failed to create dense memory dataspace");

        // HDF5 reads in row-major (C) order; Armadillo stores column-major.
        // Read into a temporary row-major buffer then transpose into the slab.
        std::vector<double> row_major_buf(static_cast<size_t>(obs_count) * static_cast<size_t>(n_var_));
        check_h5(H5Dread(dataset_id_, H5T_NATIVE_DOUBLE, mem_space, file_space,
                         H5P_DEFAULT, row_major_buf.data()) >= 0,
                 "Failed to read dense slab");

        // Copy row-major buffer into column-major Armadillo matrix.
        for (arma::uword r = 0; r < obs_count; ++r) {
            const double* row_ptr = row_major_buf.data() + static_cast<size_t>(r) * static_cast<size_t>(n_var_);
            for (arma::uword c = 0; c < n_var_; ++c) {
                slab(r, c) = row_ptr[c];
            }
        }

        H5Sclose(mem_space);
        H5Sclose(file_space);
    }

    void BackedDenseMatrixOperator::apply_transforms_(
        arma::uword obs_start, arma::mat& slab) const {

        const arma::uword nrows = slab.n_rows;
        const bool has_scale = !row_scale_.is_empty();

        if (!has_scale && !apply_log1p_) return;

        for (arma::uword r = 0; r < nrows; ++r) {
            const arma::uword obs_idx = obs_start + r;
            if (has_scale) {
                slab.row(r) *= row_scale_(obs_idx);
            }
            if (apply_log1p_) {
                for (arma::uword c = 0; c < slab.n_cols; ++c) {
                    slab(r, c) = std::log1p(slab(r, c));
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

} // namespace actionet
