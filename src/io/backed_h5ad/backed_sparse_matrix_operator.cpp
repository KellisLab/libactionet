#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/h5ad_matrix_io.hpp"

#include "_h5_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <omp.h>

namespace {
    using actionet::detail::h5::check_h5;

    constexpr size_t kSelectiveSieveBytes = 4ULL * 1024;
    // HDF5 point selection over a contiguous dataset turns into one data-sieve
    // read per touched block. At atlas scale, millions of otherwise efficient
    // 4 KiB reads lose badly to one sequential pass on cold or network-backed
    // storage. Reserve point reads for genuinely sparse hits; broader feature
    // sets use the compact index scan plus one sequential value read.
    constexpr double kPointReadMaxBlockFraction = 0.05;
    constexpr arma::uword kDirectSelectedRowLimit = 4096;
    constexpr unsigned int kMaxSelectiveScanThreads = 8;

    template <typename T>
    std::vector<T> read_integer_slice(
        hid_t dataset,
        unsigned long long start,
        unsigned long long count,
        hid_t memory_type,
        const char* error_context) {
        std::vector<T> values(static_cast<size_t>(count));
        if (count == 0) {
            return values;
        }

        const hsize_t offset[1] = {static_cast<hsize_t>(start)};
        const hsize_t extent[1] = {static_cast<hsize_t>(count)};
        actionet::detail::h5::Space file_space(H5Dget_space(dataset));
        check_h5(static_cast<bool>(file_space), "Failed to open sparse index dataspace");
        check_h5(
            H5Sselect_hyperslab(
                file_space.get(), H5S_SELECT_SET, offset, nullptr, extent, nullptr) >= 0,
            "Failed to select sparse index hyperslab");
        actionet::detail::h5::Space memory_space(H5Screate_simple(1, extent, nullptr));
        check_h5(static_cast<bool>(memory_space), "Failed to create sparse index memory space");
        check_h5(
            H5Dread(
                dataset,
                memory_type,
                memory_space.get(),
                file_space.get(),
                H5P_DEFAULT,
                values.data()) >= 0,
            error_context);
        return values;
    }

    void read_double_slice(
        hid_t dataset,
        unsigned long long start,
        unsigned long long count,
        std::vector<double>& values) {
        values.assign(static_cast<size_t>(count), 0.0);
        if (count == 0) {
            return;
        }

        const hsize_t offset[1] = {static_cast<hsize_t>(start)};
        const hsize_t extent[1] = {static_cast<hsize_t>(count)};
        actionet::detail::h5::Space file_space(H5Dget_space(dataset));
        check_h5(static_cast<bool>(file_space), "Failed to open sparse data dataspace");
        check_h5(
            H5Sselect_hyperslab(
                file_space.get(), H5S_SELECT_SET, offset, nullptr, extent, nullptr) >= 0,
            "Failed to select sparse data hyperslab");
        actionet::detail::h5::Space memory_space(H5Screate_simple(1, extent, nullptr));
        check_h5(static_cast<bool>(memory_space), "Failed to create sparse data memory space");
        check_h5(
            H5Dread(
                dataset,
                H5T_NATIVE_DOUBLE,
                memory_space.get(),
                file_space.get(),
                H5P_DEFAULT,
                values.data()) >= 0,
            "Failed to read sparse data slice");
    }

    void read_double_points(
        hid_t dataset,
        const std::vector<hsize_t>& positions,
        std::vector<double>& values) {
        values.assign(positions.size(), 0.0);
        if (positions.empty()) {
            return;
        }

        actionet::detail::h5::Space file_space(H5Dget_space(dataset));
        check_h5(static_cast<bool>(file_space), "Failed to open sparse data dataspace");
        check_h5(
            H5Sselect_elements(
                file_space.get(),
                H5S_SELECT_SET,
                positions.size(),
                positions.data()) >= 0,
            "Failed to select sparse data points");
        const hsize_t extent[1] = {static_cast<hsize_t>(positions.size())};
        actionet::detail::h5::Space memory_space(H5Screate_simple(1, extent, nullptr));
        check_h5(static_cast<bool>(memory_space), "Failed to create sparse point memory space");
        check_h5(
            H5Dread(
                dataset,
                H5T_NATIVE_DOUBLE,
                memory_space.get(),
                file_space.get(),
                H5P_DEFAULT,
                values.data()) >= 0,
            "Failed to read sparse data points");
    }
} // namespace

namespace actionet {
    BackedSparseMatrixOperator::BackedSparseMatrixOperator(
        const std::string& file_path,
        const std::string& group_path,
        arma::uword chunk_size,
        const std::vector<double>& row_scale_factors,
        bool apply_log1p,
        double log_scale,
        size_t io_target_chunk_bytes,
        double io_target_chunk_fraction_of_cap,
        int n_threads)
        : file_path_(file_path),
          group_path_(group_path),
          is_csr_(true),
          apply_log1p_(apply_log1p),
          log_scale_(log_scale),
          apply_log_scale_(apply_log1p && std::abs(log_scale - 1.0) > 0.0),
          has_row_scale_(!row_scale_factors.empty()),
          no_transform_(row_scale_factors.empty() && !apply_log1p),
          chunk_size_(std::max<arma::uword>(1, chunk_size)),
          target_chunk_nnz_(0),
          data_item_size_(0),
          indices_item_size_(0),
          indices_are_signed_(true),
          data_layout_(H5D_LAYOUT_ERROR),
          data_io_block_elements_(1),
          n_obs_(0),
          n_var_(0),
          n_threads_(static_cast<unsigned int>(std::max(0, n_threads))),
          file_id_(-1),
          group_id_(-1),
          data_ds_(-1),
          indices_ds_(-1),
          indptr_ds_(-1) {

        const auto matrix_info = h5ad::inspect_matrix(file_path_, group_path_);
        check_h5(
            matrix_info.encoding == h5ad::MatrixEncoding::CSR ||
                matrix_info.encoding == h5ad::MatrixEncoding::CSC,
            "BackedSparseMatrixOperator requires a sparse H5AD matrix");
        check_h5(
            matrix_info.rows <= std::numeric_limits<arma::uword>::max() &&
                matrix_info.cols <= std::numeric_limits<arma::uword>::max(),
            "Sparse H5AD shape exceeds the compute operator index range");
        is_csr_ = matrix_info.encoding == h5ad::MatrixEncoding::CSR;
        n_obs_ = static_cast<arma::uword>(matrix_info.rows);
        n_var_ = static_cast<arma::uword>(matrix_info.cols);

        // Disable HDF5 advisory file locking so this reader can coexist with
        // h5py/AnnData backed-mode handles that already hold a lock on the
        // same inode (errno 11 / EAGAIN from H5FD__sec2_lock otherwise).
        file_id_ = actionet::detail::h5::open_h5_readonly_no_lock(
            file_path_, "BackedSparseMatrixOperator", kSelectiveSieveBytes);

        group_id_ = H5Gopen2(file_id_, group_path_.c_str(), H5P_DEFAULT);
        check_h5(group_id_ >= 0, "Failed to open sparse matrix group path");

        data_ds_ = H5Dopen2(group_id_, "data", H5P_DEFAULT);
        indices_ds_ = H5Dopen2(group_id_, "indices", H5P_DEFAULT);
        indptr_ds_ = H5Dopen2(group_id_, "indptr", H5P_DEFAULT);
        check_h5(data_ds_ >= 0 && indices_ds_ >= 0 && indptr_ds_ >= 0,
                 "Missing sparse datasets data/indices/indptr");

        {
            actionet::detail::h5::Type data_type(H5Dget_type(data_ds_));
            actionet::detail::h5::Type indices_type(H5Dget_type(indices_ds_));
            check_h5(
                static_cast<bool>(data_type) && static_cast<bool>(indices_type),
                "Failed to inspect sparse dataset types");
            data_item_size_ = H5Tget_size(data_type.get());
            indices_item_size_ = H5Tget_size(indices_type.get());
            check_h5(data_item_size_ > 0, "Sparse data dtype has zero width");
            check_h5(
                H5Tget_class(indices_type.get()) == H5T_INTEGER &&
                    indices_item_size_ > 0 && indices_item_size_ <= 8,
                "Sparse indices must use an integer dtype no wider than 64 bits");
            indices_are_signed_ = H5Tget_sign(indices_type.get()) != H5T_SGN_NONE;

            actionet::detail::h5::Property dcpl(H5Dget_create_plist(data_ds_));
            check_h5(static_cast<bool>(dcpl), "Failed to inspect sparse data layout");
            data_layout_ = H5Pget_layout(dcpl.get());
            if (data_layout_ == H5D_CHUNKED) {
                hsize_t chunk_extent[1] = {0};
                check_h5(
                    H5Pget_chunk(dcpl.get(), 1, chunk_extent) == 1 &&
                        chunk_extent[0] > 0,
                    "Sparse data must use one-dimensional HDF5 chunks");
                data_io_block_elements_ = chunk_extent[0];
            } else if (data_layout_ == H5D_CONTIGUOUS) {
                data_io_block_elements_ = static_cast<hsize_t>(
                    std::max<size_t>(1, kSelectiveSieveBytes / data_item_size_));
            } else {
                data_io_block_elements_ = std::numeric_limits<hsize_t>::max();
            }
        }

        hid_t indptr_space = H5Dget_space(indptr_ds_);
        check_h5(indptr_space >= 0, "Failed to get indptr dataspace");
        check_h5(H5Sget_simple_extent_ndims(indptr_space) == 1, "indptr must be 1D");
        hsize_t indptr_dim[1] = {0};
        check_h5(H5Sget_simple_extent_dims(indptr_space, indptr_dim, nullptr) == 1,
                 "Failed to get indptr dimensions");
        indptr_.assign(static_cast<size_t>(indptr_dim[0]), 0ULL);
        if (!indptr_.empty()) {
            // Read indptr using H5T_NATIVE_LLONG (signed 64-bit) to handle both
            // int32-stored (common for NNZ < 2^31) and int64-stored (large matrices)
            // indptr arrays. HDF5 converts int32 -> int64 automatically.
            // We then copy into the uint64 indptr_ vector, which is safe because
            // valid indptr values are always non-negative.
            std::vector<long long> indptr_signed(static_cast<size_t>(indptr_dim[0]), 0LL);
            check_h5(H5Dread(indptr_ds_, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                             indptr_signed.data()) >= 0,
                     "Failed to read indptr");
            for (size_t i = 0; i < indptr_signed.size(); ++i) {
                check_h5(indptr_signed[i] >= 0, "indptr contains negative value");
                indptr_[i] = static_cast<unsigned long long>(indptr_signed[i]);
            }
        }
        H5Sclose(indptr_space);

        const size_t expected = static_cast<size_t>((is_csr_ ? n_obs_ : n_var_) + 1);
        check_h5(indptr_.size() == expected, "indptr length does not match sparse shape");

        const size_t bytes_per_nnz = sizeof(double) + sizeof(unsigned long long);
        if (io_target_chunk_bytes > 0) {
            target_chunk_nnz_ = static_cast<unsigned long long>(
                std::max<size_t>(1, io_target_chunk_bytes / bytes_per_nnz));
        } else {
            check_h5(std::isfinite(io_target_chunk_fraction_of_cap) && io_target_chunk_fraction_of_cap > 0.0,
                     "io_target_chunk_fraction_of_cap must be finite and > 0 when io_target_chunk_bytes == 0");
            const unsigned long long total_nnz = indptr_.empty() ? 0ULL : indptr_.back();
            const arma::uword axis_len = is_csr_ ? n_obs_ : n_var_;
            if (total_nnz > 0ULL && axis_len > 0) {
                const double mean_nnz_per_axis_entry =
                    static_cast<double>(total_nnz) / static_cast<double>(axis_len);
                const double estimated_full_chunk_nnz =
                    static_cast<double>(chunk_size_) * mean_nnz_per_axis_entry;
                const double auto_target_nnz =
                    std::max<double>(1.0, std::ceil(
                        io_target_chunk_fraction_of_cap * estimated_full_chunk_nnz));
                target_chunk_nnz_ = static_cast<unsigned long long>(auto_target_nnz);
            }
        }

        if (!row_scale_factors.empty()) {
            check_h5(row_scale_factors.size() == static_cast<size_t>(n_obs_),
                     "row_scale_factors length must equal n_obs");
            row_scale_ = arma::vec(row_scale_factors);
        }
        check_h5(std::isfinite(log_scale_) && log_scale_ > 0.0,
                 "log_scale must be finite and > 0");
    }

    void BackedSparseMatrixOperator::close_handles_() {
        if (indptr_ds_ >= 0) { H5Dclose(indptr_ds_); indptr_ds_ = -1; }
        if (indices_ds_ >= 0) { H5Dclose(indices_ds_); indices_ds_ = -1; }
        if (data_ds_ >= 0) { H5Dclose(data_ds_); data_ds_ = -1; }
        if (group_id_ >= 0) { H5Gclose(group_id_); group_id_ = -1; }
        if (file_id_ >= 0) { H5Fclose(file_id_); file_id_ = -1; }
    }

    BackedSparseMatrixOperator::~BackedSparseMatrixOperator() {
        close_handles_();
    }

    BackedSparseMatrixOperator::BackedSparseMatrixOperator(BackedSparseMatrixOperator&& other) noexcept
        : file_path_(std::move(other.file_path_)),
          group_path_(std::move(other.group_path_)),
          is_csr_(other.is_csr_),
          apply_log1p_(other.apply_log1p_),
          log_scale_(other.log_scale_),
          apply_log_scale_(other.apply_log_scale_),
          has_row_scale_(other.has_row_scale_),
          no_transform_(other.no_transform_),
          chunk_size_(other.chunk_size_),
          target_chunk_nnz_(other.target_chunk_nnz_),
          data_item_size_(other.data_item_size_),
          indices_item_size_(other.indices_item_size_),
          indices_are_signed_(other.indices_are_signed_),
          data_layout_(other.data_layout_),
          data_io_block_elements_(other.data_io_block_elements_),
          n_obs_(other.n_obs_),
          n_var_(other.n_var_),
          row_scale_(std::move(other.row_scale_)),
          n_threads_(other.n_threads_),
          file_id_(other.file_id_),
          group_id_(other.group_id_),
          data_ds_(other.data_ds_),
          indices_ds_(other.indices_ds_),
          indptr_ds_(other.indptr_ds_),
          indptr_(std::move(other.indptr_)),
          chunk_cache_(std::move(other.chunk_cache_)) {
        other.file_id_ = -1;
        other.group_id_ = -1;
        other.data_ds_ = -1;
        other.indices_ds_ = -1;
        other.indptr_ds_ = -1;
    }

    BackedSparseMatrixOperator& BackedSparseMatrixOperator::operator=(BackedSparseMatrixOperator&& other) noexcept {
        if (this != &other) {
            close_handles_();
            file_path_ = std::move(other.file_path_);
            group_path_ = std::move(other.group_path_);
            is_csr_ = other.is_csr_;
            apply_log1p_ = other.apply_log1p_;
            log_scale_ = other.log_scale_;
            apply_log_scale_ = other.apply_log_scale_;
            has_row_scale_ = other.has_row_scale_;
            no_transform_ = other.no_transform_;
            chunk_size_ = other.chunk_size_;
            target_chunk_nnz_ = other.target_chunk_nnz_;
            data_item_size_ = other.data_item_size_;
            indices_item_size_ = other.indices_item_size_;
            indices_are_signed_ = other.indices_are_signed_;
            data_layout_ = other.data_layout_;
            data_io_block_elements_ = other.data_io_block_elements_;
            n_obs_ = other.n_obs_;
            n_var_ = other.n_var_;
            row_scale_ = std::move(other.row_scale_);
            n_threads_ = other.n_threads_;
            file_id_ = other.file_id_;
            group_id_ = other.group_id_;
            data_ds_ = other.data_ds_;
            indices_ds_ = other.indices_ds_;
            indptr_ds_ = other.indptr_ds_;
            indptr_ = std::move(other.indptr_);
            chunk_cache_ = std::move(other.chunk_cache_);
            other.file_id_ = -1;
            other.group_id_ = -1;
            other.data_ds_ = -1;
            other.indices_ds_ = -1;
            other.indptr_ds_ = -1;
        }
        return *this;
    }

    void BackedSparseMatrixOperator::read_data_indices_slice_(
        unsigned long long start, unsigned long long count,
        std::vector<double>& data, std::vector<unsigned long long>& indices) const {
        data.assign(static_cast<size_t>(count), 0.0);
        indices.assign(static_cast<size_t>(count), 0ULL);
        if (count == 0) {
            return;
        }

        hsize_t start_h[1] = {static_cast<hsize_t>(start)};
        hsize_t count_h[1] = {static_cast<hsize_t>(count)};

        hid_t file_space_data = H5Dget_space(data_ds_);
        check_h5(file_space_data >= 0, "Failed to get data dataspace");
        check_h5(H5Sselect_hyperslab(file_space_data, H5S_SELECT_SET, start_h, nullptr, count_h, nullptr) >= 0,
                 "Failed to select data hyperslab");
        hid_t mem_space_data = H5Screate_simple(1, count_h, nullptr);
        check_h5(mem_space_data >= 0, "Failed to create data memory dataspace");
        check_h5(H5Dread(data_ds_, H5T_NATIVE_DOUBLE, mem_space_data, file_space_data, H5P_DEFAULT, data.data()) >= 0,
                 "Failed to read sparse data slice");
        H5Sclose(mem_space_data);
        H5Sclose(file_space_data);

        hid_t file_space_indices = H5Dget_space(indices_ds_);
        check_h5(file_space_indices >= 0, "Failed to get indices dataspace");
        check_h5(H5Sselect_hyperslab(file_space_indices, H5S_SELECT_SET, start_h, nullptr, count_h, nullptr) >= 0,
                 "Failed to select indices hyperslab");
        hid_t mem_space_indices = H5Screate_simple(1, count_h, nullptr);
        check_h5(mem_space_indices >= 0, "Failed to create indices memory dataspace");
        // Read with H5T_NATIVE_LLONG to handle both int32 and int64 stored indices.
        // HDF5 converts int32 -> int64 automatically; we re-cast to uint64 after.
        std::vector<long long> indices_signed(static_cast<size_t>(count), 0LL);
        check_h5(H5Dread(indices_ds_, H5T_NATIVE_LLONG, mem_space_indices, file_space_indices, H5P_DEFAULT,
                         indices_signed.data()) >= 0,
                 "Failed to read sparse indices slice");
        for (size_t i = 0; i < indices_signed.size(); ++i) {
            indices[i] = static_cast<unsigned long long>(indices_signed[i]);
        }
        H5Sclose(mem_space_indices);
        H5Sclose(file_space_indices);
    }

    void BackedSparseMatrixOperator::load_chunk_cached_(
        unsigned long long start, unsigned long long count,
        const std::vector<double>*& data, const std::vector<unsigned long long>*& indices) const {
        if (!(chunk_cache_.count == count && chunk_cache_.start == start && count > 0)) {
            read_data_indices_slice_(start, count, chunk_cache_.data, chunk_cache_.indices);
            chunk_cache_.start = start;
            chunk_cache_.count = count;
            chunk_cache_.transformed = false;
        }
        data = &chunk_cache_.data;
        indices = &chunk_cache_.indices;
    }

    void BackedSparseMatrixOperator::ensure_chunk_transformed_csr_(
        arma::uword row_start, arma::uword row_end,
        unsigned long long nnz_start) const {
        if (no_transform_ || chunk_cache_.transformed) return;
        auto& data = chunk_cache_.data;
        for (arma::uword r = row_start; r < row_end; ++r) {
            const double rscale = row_scale_for_(r);
            const auto ls = static_cast<size_t>(indptr_[r] - nnz_start);
            const auto le = static_cast<size_t>(indptr_[r + 1] - nnz_start);
            for (size_t p = ls; p < le; ++p) {
                data[p] = transform_scaled_(data[p] * rscale);
            }
        }
        chunk_cache_.transformed = true;
    }

    void BackedSparseMatrixOperator::ensure_chunk_transformed_csc_() const {
        if (no_transform_ || chunk_cache_.transformed) return;
        auto& data = chunk_cache_.data;
        const auto& indices = chunk_cache_.indices;
        for (size_t p = 0; p < data.size(); ++p) {
            const arma::uword row = static_cast<arma::uword>(indices[p]);
            data[p] = transform_value_(row, data[p]);
        }
        chunk_cache_.transformed = true;
    }

    arma::uword BackedSparseMatrixOperator::next_block_end_(arma::uword start, arma::uword limit) const {
        const arma::uword hard_end = std::min<arma::uword>(limit, start + chunk_size_);
        if (target_chunk_nnz_ == 0 || hard_end <= start + 1) {
            return hard_end;
        }

        const unsigned long long base = indptr_[start];
        arma::uword end = start + 1;
        while (end < hard_end) {
            const unsigned long long nnz = indptr_[end] - base;
            if (nnz >= target_chunk_nnz_) {
                break;
            }
            ++end;
        }
        return std::max<arma::uword>(start + 1, end);
    }

    void BackedSparseMatrixOperator::matvec(const arma::vec& x, arma::vec& y) const {
        // S is n_obs × n_var.  matvec: y = S * x  (x n_var-length, y n_obs-length).
        if (x.n_elem != n_var_) {
            throw std::runtime_error("BackedSparseMatrixOperator::matvec dimension mismatch");
        }
        if (is_csr_) {
            matvec_csr_impl_(x, y);
        } else {
            matvec_csc_impl_(x, y);
        }
    }

    void BackedSparseMatrixOperator::rmatvec(const arma::vec& x, arma::vec& y) const {
        // S is n_obs × n_var.  rmatvec: y = S' * x  (x n_obs-length, y n_var-length).
        if (x.n_elem != n_obs_) {
            throw std::runtime_error("BackedSparseMatrixOperator::rmatvec dimension mismatch");
        }
        if (is_csr_) {
            rmatvec_csr_impl_(x, y);
        } else {
            rmatvec_csc_impl_(x, y);
        }
    }

    void BackedSparseMatrixOperator::matmat(const arma::mat& X, arma::mat& Y) const {
        // S is n_obs × n_var.  matmat: Y = S * X  (X (n_var × q) -> Y (n_obs × q)).
        if (X.n_rows != n_var_) {
            throw std::runtime_error("BackedSparseMatrixOperator::matmat dimension mismatch");
        }
        if (is_csr_) {
            matmat_csr_impl_(X, Y);
        } else {
            matmat_csc_impl_(X, Y);
        }
    }

    void BackedSparseMatrixOperator::rmatmat(const arma::mat& X, arma::mat& Y) const {
        // S is n_obs × n_var.  rmatmat: Y = S' * X  (X (n_obs × q) -> Y (n_var × q)).
        if (X.n_rows != n_obs_) {
            throw std::runtime_error("BackedSparseMatrixOperator::rmatmat dimension mismatch");
        }
        if (is_csr_) {
            rmatmat_csr_impl_(X, Y);
        } else {
            rmatmat_csc_impl_(X, Y);
        }
    }

    // ---------------------------------------------------------------------
    // CSR kernels — semantic mapping to storage layout
    // ---------------------------------------------------------------------
    //
    // For CSR storage (indptr indexes n_obs rows), the natural single pass
    // touches every stored (r, c) exactly once by scanning rows in order:
    //
    //   for r in [0, n_obs):
    //     for p in [indptr[r], indptr[r+1]):
    //       c = indices[p]; v = data[p]
    //       ...update y with (r, c, v)...
    //
    // Two useful accumulations fall out:
    //   * y[r] += v * x[c]    → y = S * x   (matvec)
    //   * y[c] += v * x[r]    → y = S' * x  (rmatvec)
    //
    // ``rmatvec_csr_impl_`` is therefore the storage-native walk; matvec on
    // CSR requires per-row accumulation into a scalar before writing y[r].

    void BackedSparseMatrixOperator::rmatvec_csr_impl_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_var_);

        for (arma::uword row_start = 0; row_start < n_obs_;) {
            const arma::uword row_end = next_block_end_(row_start, n_obs_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csr_(row_start, row_end, nnz_start);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long local_start = indptr_[r] - nnz_start;
                const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                const double xval = x(r);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword col = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                    y(col) += (*data)[static_cast<size_t>(p)] * xval;
                }
            }
            row_start = row_end;
        }
    }

    void BackedSparseMatrixOperator::matvec_csr_impl_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_obs_);

        for (arma::uword row_start = 0; row_start < n_obs_;) {
            const arma::uword row_end = next_block_end_(row_start, n_obs_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csr_(row_start, row_end, nnz_start);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long local_start = indptr_[r] - nnz_start;
                const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                double acc = 0.0;
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword col = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                    acc += (*data)[static_cast<size_t>(p)] * x(col);
                }
                y(r) = acc;
            }
            row_start = row_end;
        }
    }

    void BackedSparseMatrixOperator::rmatmat_csr_impl_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_var_, X.n_cols);
        const arma::uword q = X.n_cols;
        const unsigned int threads_use = actionet::get_num_threads_nested_safe(static_cast<unsigned int>(q), n_threads_);

        for (arma::uword row_start = 0; row_start < n_obs_;) {
            const arma::uword row_end = next_block_end_(row_start, n_obs_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csr_(row_start, row_end, nnz_start);

            #pragma omp parallel for schedule(static) num_threads(threads_use) if(threads_use > 1 && q > 1)
            for (arma::sword js = 0; js < static_cast<arma::sword>(q); ++js) {
                const arma::uword j = static_cast<arma::uword>(js);
                const double* x_col = X.colptr(j);
                double* y_col = Y.colptr(j);
                for (arma::uword r = row_start; r < row_end; ++r) {
                    const unsigned long long local_start = indptr_[r] - nnz_start;
                    const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                    const double xval = x_col[r];
                    for (unsigned long long p = local_start; p < local_end; ++p) {
                        const arma::uword col = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                        y_col[col] += (*data)[static_cast<size_t>(p)] * xval;
                    }
                }
            }
            row_start = row_end;
        }
    }

    void BackedSparseMatrixOperator::matmat_csr_impl_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_obs_, X.n_cols);
        const arma::uword q = X.n_cols;
        const unsigned int threads_use = actionet::get_num_threads_nested_safe(static_cast<unsigned int>(q), n_threads_);

        for (arma::uword row_start = 0; row_start < n_obs_;) {
            const arma::uword row_end = next_block_end_(row_start, n_obs_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csr_(row_start, row_end, nnz_start);

            #pragma omp parallel for schedule(static) num_threads(threads_use) if(threads_use > 1 && q > 1)
            for (arma::sword js = 0; js < static_cast<arma::sword>(q); ++js) {
                const arma::uword j = static_cast<arma::uword>(js);
                const double* x_col = X.colptr(j);
                double* y_col = Y.colptr(j);
                for (arma::uword r = row_start; r < row_end; ++r) {
                    const unsigned long long local_start = indptr_[r] - nnz_start;
                    const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                    double acc = 0.0;
                    for (unsigned long long p = local_start; p < local_end; ++p) {
                        const arma::uword col = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                        acc += (*data)[static_cast<size_t>(p)] * x_col[col];
                    }
                    y_col[r] += acc;
                }
            }
            row_start = row_end;
        }
    }

    // ---------------------------------------------------------------------
    // CSC kernels — semantic mapping to storage layout
    // ---------------------------------------------------------------------
    //
    // For CSC storage (indptr indexes n_var columns), the natural single
    // pass walks each column's contiguous NNZ run:
    //
    //   for c in [0, n_var):
    //     for p in [indptr[c], indptr[c+1]):
    //       r = indices[p]; v = data[p]
    //       ...update y with (r, c, v)...
    //
    // The natural accumulations are the mirror of the CSR case:
    //   * y[c] += v * x[r]    → y = S' * x  (rmatvec, per-column dot)
    //   * y[r] += v * x[c]    → y = S  * x  (matvec, scatter into rows)
    //
    // As for CSR, ``rmatvec_csc_impl_`` is therefore the storage-native
    // walk; matvec on CSC scatters into y[r] across the same NNZ pass.

    void BackedSparseMatrixOperator::rmatvec_csc_impl_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_var_);

        for (arma::uword col_start = 0; col_start < n_var_;) {
            const arma::uword col_end = next_block_end_(col_start, n_var_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csc_();

            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long local_start = indptr_[c] - nnz_start;
                const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                double acc = 0.0;
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword row = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                    acc += (*data)[static_cast<size_t>(p)] * x(row);
                }
                y(c) = acc;
            }
            col_start = col_end;
        }
    }

    void BackedSparseMatrixOperator::matvec_csc_impl_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_obs_);

        for (arma::uword col_start = 0; col_start < n_var_;) {
            const arma::uword col_end = next_block_end_(col_start, n_var_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csc_();

            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long local_start = indptr_[c] - nnz_start;
                const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                const double xval = x(c);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword row = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                    y(row) += (*data)[static_cast<size_t>(p)] * xval;
                }
            }
            col_start = col_end;
        }
    }

    void BackedSparseMatrixOperator::rmatmat_csc_impl_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_var_, X.n_cols);
        const arma::uword q = X.n_cols;
        const unsigned int threads_use = actionet::get_num_threads_nested_safe(static_cast<unsigned int>(q), n_threads_);

        for (arma::uword col_start = 0; col_start < n_var_;) {
            const arma::uword col_end = next_block_end_(col_start, n_var_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csc_();

            #pragma omp parallel for schedule(static) num_threads(threads_use) if(threads_use > 1 && q > 1)
            for (arma::sword js = 0; js < static_cast<arma::sword>(q); ++js) {
                const arma::uword j = static_cast<arma::uword>(js);
                const double* x_col = X.colptr(j);
                double* y_col = Y.colptr(j);
                for (arma::uword c = col_start; c < col_end; ++c) {
                    const unsigned long long local_start = indptr_[c] - nnz_start;
                    const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                    double acc = 0.0;
                    for (unsigned long long p = local_start; p < local_end; ++p) {
                        const arma::uword row = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                        acc += (*data)[static_cast<size_t>(p)] * x_col[row];
                    }
                    y_col[c] += acc;
                }
            }
            col_start = col_end;
        }
    }

    void BackedSparseMatrixOperator::matmat_csc_impl_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_obs_, X.n_cols);
        const arma::uword q = X.n_cols;
        const unsigned int threads_use = actionet::get_num_threads_nested_safe(static_cast<unsigned int>(q), n_threads_);

        for (arma::uword col_start = 0; col_start < n_var_;) {
            const arma::uword col_end = next_block_end_(col_start, n_var_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            const std::vector<double>* data;
            const std::vector<unsigned long long>* indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);
            ensure_chunk_transformed_csc_();

            #pragma omp parallel for schedule(static) num_threads(threads_use) if(threads_use > 1 && q > 1)
            for (arma::sword js = 0; js < static_cast<arma::sword>(q); ++js) {
                const arma::uword j = static_cast<arma::uword>(js);
                const double* x_col = X.colptr(j);
                double* y_col = Y.colptr(j);
                for (arma::uword c = col_start; c < col_end; ++c) {
                    const unsigned long long local_start = indptr_[c] - nnz_start;
                    const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                    const double xval = x_col[c];
                    for (unsigned long long p = local_start; p < local_end; ++p) {
                        const arma::uword row = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                        y_col[row] += (*data)[static_cast<size_t>(p)] * xval;
                    }
                }
            }
            col_start = col_end;
        }
    }

    // ---- rowStats implementation ------------------------------------------------

    void BackedSparseMatrixOperator::rowStats(
        arma::vec& row_sum, arma::vec& row_sum_sq, arma::vec& nnz) const {
        if (is_csr_) {
            row_stats_csr_(row_sum, row_sum_sq, nnz);
        } else {
            row_stats_csc_(row_sum, row_sum_sq, nnz);
        }
    }

    void BackedSparseMatrixOperator::row_stats_csr_(
        arma::vec& row_sum, arma::vec& row_sum_sq, arma::vec& nnz_out) const {
        row_sum.zeros(n_obs_);
        row_sum_sq.zeros(n_obs_);
        nnz_out.zeros(n_obs_);

        for (arma::uword row_start = 0; row_start < n_obs_;) {
            const arma::uword row_end = next_block_end_(row_start, n_obs_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            if (nnz_count > 0) {
                const std::vector<double>* data;
                const std::vector<unsigned long long>* indices;
                load_chunk_cached_(nnz_start, nnz_count, data, indices);
                ensure_chunk_transformed_csr_(row_start, row_end, nnz_start);

                for (arma::uword r = row_start; r < row_end; ++r) {
                    const unsigned long long local_start = indptr_[r] - nnz_start;
                    const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                    double rs = 0.0, rssq = 0.0;
                    double cnt = 0.0;
                    for (unsigned long long p = local_start; p < local_end; ++p) {
                        const double v = (*data)[static_cast<size_t>(p)];
                        rs += v;
                        rssq += v * v;
                        cnt += 1.0;
                    }
                    row_sum(r) = rs;
                    row_sum_sq(r) = rssq;
                    nnz_out(r) = cnt;
                }
            }
            row_start = row_end;
        }
    }

    void BackedSparseMatrixOperator::row_stats_csc_(
        arma::vec& row_sum, arma::vec& row_sum_sq, arma::vec& nnz_out) const {
        row_sum.zeros(n_obs_);
        row_sum_sq.zeros(n_obs_);
        nnz_out.zeros(n_obs_);

        for (arma::uword col_start = 0; col_start < n_var_;) {
            const arma::uword col_end = next_block_end_(col_start, n_var_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            if (nnz_count > 0) {
                const std::vector<double>* data;
                const std::vector<unsigned long long>* indices;
                load_chunk_cached_(nnz_start, nnz_count, data, indices);
                ensure_chunk_transformed_csc_();

                for (unsigned long long p = 0; p < nnz_count; ++p) {
                    const arma::uword row = static_cast<arma::uword>(
                        (*indices)[static_cast<size_t>(p)]);
                    const double v = (*data)[static_cast<size_t>(p)];
                    row_sum(row) += v;
                    row_sum_sq(row) += v * v;
                    nnz_out(row) += 1.0;
                }
            }
            col_start = col_end;
        }
    }

    // ---- takeColumns implementations ------------------------------------------------

    void BackedSparseMatrixOperator::take_columns_dense_csr_(
        const arma::uvec& col_indices,
        const arma::uvec& row_indices,
        arma::mat& out) const {

        const arma::uword n_sel_cols = col_indices.n_elem;
        const bool all_rows = row_indices.is_empty();
        const arma::uword n_sel_rows = all_rows ? n_obs_ : row_indices.n_elem;

        out.zeros(n_sel_rows, n_sel_cols);
        if (n_sel_cols == 0 || n_sel_rows == 0) {
            return;
        }

        // Build col -> output-position lookup (sentinel = n_sel_cols means "skip").
        std::vector<arma::uword> col_map(static_cast<size_t>(n_var_), n_sel_cols);
        for (arma::uword j = 0; j < n_sel_cols; ++j) {
            const arma::uword c = col_indices(j);
            if (col_map[c] == n_sel_cols) {
                col_map[c] = j;
            }
        }

        struct Match {
            hsize_t position;
            arma::uword source_row;
            arma::uword output_row;
            arma::uword output_col;
        };

        auto read_matches = [&](arma::uword row_start,
                                arma::uword row_end,
                                const std::vector<arma::uword>* row_map,
                                arma::uword missing_row,
                                std::vector<Match>& matches) {
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;
            matches.clear();
            if (nnz_count == 0) {
                return;
            }

            auto collect = [&](const auto& raw_indices) {
                using RawIndex = typename std::decay_t<decltype(raw_indices)>::value_type;
                const arma::uword chunk_rows = row_end - row_start;
                const unsigned int available_threads =
                    actionet::get_num_threads_nested_safe(
                        static_cast<unsigned int>(chunk_rows), n_threads_);
                const unsigned int threads_use = std::max(
                    1U,
                    std::min(kMaxSelectiveScanThreads, available_threads));
                std::vector<std::vector<Match>> thread_matches(threads_use);
                std::atomic<bool> invalid_index{false};

                #pragma omp parallel num_threads(threads_use) if(threads_use > 1)
                {
                    const unsigned int thread_id =
                        static_cast<unsigned int>(omp_get_thread_num());
                    auto& local_matches = thread_matches[thread_id];

                    #pragma omp for schedule(static)
                    for (arma::sword rs = static_cast<arma::sword>(row_start);
                         rs < static_cast<arma::sword>(row_end);
                         ++rs) {
                        const arma::uword r = static_cast<arma::uword>(rs);
                        const arma::uword output_row =
                            row_map == nullptr
                                ? r
                                : (*row_map)[static_cast<size_t>(r)];
                        if (output_row == missing_row) {
                            continue;
                        }
                        const unsigned long long local_start =
                            indptr_[r] - nnz_start;
                        const unsigned long long local_end =
                            indptr_[r + 1] - nnz_start;
                        for (unsigned long long p = local_start;
                             p < local_end;
                             ++p) {
                            const RawIndex raw =
                                raw_indices[static_cast<size_t>(p)];
                            if constexpr (std::is_signed_v<RawIndex>) {
                                if (raw < 0) {
                                    invalid_index.store(
                                        true, std::memory_order_relaxed);
                                    continue;
                                }
                            }
                            const auto as_unsigned =
                                static_cast<unsigned long long>(raw);
                            if (as_unsigned >=
                                static_cast<unsigned long long>(n_var_)) {
                                invalid_index.store(
                                    true, std::memory_order_relaxed);
                                continue;
                            }
                            const arma::uword output_col =
                                col_map[static_cast<size_t>(as_unsigned)];
                            if (output_col != n_sel_cols) {
                                local_matches.push_back(Match{
                                    static_cast<hsize_t>(nnz_start + p),
                                    r,
                                    output_row,
                                    output_col});
                            }
                        }
                    }
                }

                check_h5(
                    !invalid_index.load(std::memory_order_relaxed),
                    "Sparse column index is negative or exceeds matrix shape");
                size_t total_matches = 0;
                for (const auto& local_matches : thread_matches) {
                    total_matches += local_matches.size();
                }
                matches.reserve(total_matches);
                for (auto& local_matches : thread_matches) {
                    matches.insert(
                        matches.end(),
                        std::make_move_iterator(local_matches.begin()),
                        std::make_move_iterator(local_matches.end()));
                }
            };

            if (indices_item_size_ <= 4) {
                if (indices_are_signed_) {
                    collect(read_integer_slice<std::int32_t>(
                        indices_ds_,
                        nnz_start,
                        nnz_count,
                        H5T_NATIVE_INT32,
                        "Failed to read compact signed sparse indices"));
                } else {
                    collect(read_integer_slice<std::uint32_t>(
                        indices_ds_,
                        nnz_start,
                        nnz_count,
                        H5T_NATIVE_UINT32,
                        "Failed to read compact unsigned sparse indices"));
                }
            } else if (indices_are_signed_) {
                collect(read_integer_slice<std::int64_t>(
                    indices_ds_,
                    nnz_start,
                    nnz_count,
                    H5T_NATIVE_INT64,
                    "Failed to read signed sparse indices"));
            } else {
                collect(read_integer_slice<std::uint64_t>(
                    indices_ds_,
                    nnz_start,
                    nnz_count,
                    H5T_NATIVE_UINT64,
                    "Failed to read unsigned sparse indices"));
            }
        };

        auto materialize_matches = [&](const std::vector<Match>& matches,
                                       unsigned long long nnz_start,
                                       unsigned long long nnz_end) {
            if (matches.empty()) {
                return;
            }

            const hsize_t first_block =
                matches.front().position / data_io_block_elements_;
            hsize_t previous_block = first_block;
            unsigned long long touched_blocks = 1;
            for (size_t i = 1; i < matches.size(); ++i) {
                const hsize_t block =
                    matches[i].position / data_io_block_elements_;
                if (block != previous_block) {
                    ++touched_blocks;
                    previous_block = block;
                }
            }

            const hsize_t source_first_block =
                static_cast<hsize_t>(nnz_start) / data_io_block_elements_;
            const hsize_t source_last_block =
                static_cast<hsize_t>(nnz_end - 1) / data_io_block_elements_;
            const unsigned long long source_blocks =
                static_cast<unsigned long long>(
                    source_last_block - source_first_block + 1);
            const bool use_points =
                data_layout_ != H5D_COMPACT &&
                static_cast<double>(touched_blocks) <=
                    kPointReadMaxBlockFraction *
                        static_cast<double>(source_blocks);

            std::vector<double> values;
            if (use_points) {
                std::vector<hsize_t> positions;
                positions.reserve(matches.size());
                for (const Match& match : matches) {
                    positions.push_back(match.position);
                }
                read_double_points(data_ds_, positions, values);
                for (size_t i = 0; i < matches.size(); ++i) {
                    const Match& match = matches[i];
                    out(match.output_row, match.output_col) =
                        transform_value_(match.source_row, values[i]);
                }
            } else {
                read_double_slice(data_ds_, nnz_start, nnz_end - nnz_start, values);
                for (const Match& match : matches) {
                    out(match.output_row, match.output_col) =
                        transform_value_(
                            match.source_row,
                            values[static_cast<size_t>(
                                match.position - static_cast<hsize_t>(nnz_start))]);
                }
            }
        };

        if (all_rows) {
            std::vector<Match> matches;
            for (arma::uword row_start = 0; row_start < n_obs_;) {
                const arma::uword row_end = next_block_end_(row_start, n_obs_);
                const unsigned long long nnz_start = indptr_[row_start];
                const unsigned long long nnz_end = indptr_[row_end];
                read_matches(
                    row_start,
                    row_end,
                    nullptr,
                    n_obs_,
                    matches);
                materialize_matches(matches, nnz_start, nnz_end);
                row_start = row_end;
            }
        } else if (n_sel_rows <= kDirectSelectedRowLimit) {
            // Sparse row requests should not pay for a complete CSR scan.
            // Sort request slots by source row, read each unique row once,
            // and copy it into duplicate output slots in request order.
            std::vector<arma::uword> order(static_cast<size_t>(n_sel_rows));
            std::iota(order.begin(), order.end(), arma::uword{0});
            std::stable_sort(
                order.begin(),
                order.end(),
                [&](arma::uword lhs, arma::uword rhs) {
                    return row_indices(lhs) < row_indices(rhs);
                });

            std::vector<Match> matches;
            size_t cursor = 0;
            while (cursor < order.size()) {
                const arma::uword source_row = row_indices(order[cursor]);
                const arma::uword first_output_row = order[cursor];
                read_matches(
                    source_row,
                    source_row + 1,
                    nullptr,
                    n_obs_,
                    matches);
                for (Match& match : matches) {
                    match.output_row = first_output_row;
                }
                materialize_matches(
                    matches,
                    indptr_[source_row],
                    indptr_[source_row + 1]);

                size_t next = cursor + 1;
                while (next < order.size() &&
                       row_indices(order[next]) == source_row) {
                    out.row(order[next]) = out.row(first_output_row);
                    ++next;
                }
                cursor = next;
            }
        } else {
            // Large arbitrary row requests amortize better through one full
            // index scan. Keep only the first output slot in the lookup and
            // restore duplicate rows after the scan.
            std::vector<arma::uword> row_map(
                static_cast<size_t>(n_obs_), n_sel_rows);
            for (arma::uword i = 0; i < n_sel_rows; ++i) {
                if (row_map[static_cast<size_t>(row_indices(i))] == n_sel_rows) {
                    row_map[static_cast<size_t>(row_indices(i))] = i;
                }
            }

            std::vector<Match> matches;
            for (arma::uword row_start = 0; row_start < n_obs_;) {
                const arma::uword row_end = next_block_end_(row_start, n_obs_);
                const unsigned long long nnz_start = indptr_[row_start];
                const unsigned long long nnz_end = indptr_[row_end];
                read_matches(
                    row_start,
                    row_end,
                    &row_map,
                    n_sel_rows,
                    matches);
                materialize_matches(matches, nnz_start, nnz_end);
                row_start = row_end;
            }

            for (arma::uword i = 0; i < n_sel_rows; ++i) {
                const arma::uword first =
                    row_map[static_cast<size_t>(row_indices(i))];
                if (first != i) {
                    out.row(i) = out.row(first);
                }
            }
        }

        // Handle duplicate columns: copy first-match values.
        for (arma::uword j = 0; j < n_sel_cols; ++j) {
            const arma::uword c = col_indices(j);
            if (col_map[c] != j) {
                out.col(j) = out.col(col_map[c]);
            }
        }
    }

    void BackedSparseMatrixOperator::take_columns_dense_csc_(
        const arma::uvec& col_indices,
        const arma::uvec& row_indices,
        arma::mat& out) const {

        const arma::uword n_sel_cols = col_indices.n_elem;
        const bool all_rows = row_indices.is_empty();
        const arma::uword n_sel_rows = all_rows ? n_obs_ : row_indices.n_elem;

        out.zeros(n_sel_rows, n_sel_cols);

        // Build row -> output-position lookup when row_indices is given.
        std::vector<arma::uword> row_map;
        if (!all_rows) {
            row_map.assign(static_cast<size_t>(n_obs_), n_sel_rows);
            for (arma::uword i = 0; i < n_sel_rows; ++i) {
                const arma::uword r = row_indices(i);
                if (row_map[r] == n_sel_rows) {
                    row_map[r] = i;
                }
            }
        }

        // CSC: iterate directly over the requested columns' indptr ranges.
        for (arma::uword j = 0; j < n_sel_cols; ++j) {
            const arma::uword c = col_indices(j);
            const unsigned long long nnz_start = indptr_[c];
            const unsigned long long nnz_end = indptr_[c + 1];
            const unsigned long long nnz_count = nnz_end - nnz_start;
            if (nnz_count == 0) continue;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            read_data_indices_slice_(nnz_start, nnz_count, data, indices);

            for (unsigned long long p = 0; p < nnz_count; ++p) {
                const arma::uword row = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                const arma::uword out_row = all_rows ? row : row_map[row];
                if (out_row == n_sel_rows) continue;
                out(out_row, j) = transform_value_(row, data[static_cast<size_t>(p)]);
            }
        }

        if (!all_rows) {
            for (arma::uword i = 0; i < n_sel_rows; ++i) {
                const arma::uword first = row_map[row_indices(i)];
                if (first != i) {
                    out.row(i) = out.row(first);
                }
            }
        }
    }

    arma::mat BackedSparseMatrixOperator::takeColumnsDense(
        const arma::uvec& col_indices,
        const arma::uvec& row_indices) const {

        for (arma::uword i = 0; i < col_indices.n_elem; ++i) {
            if (col_indices(i) >= n_var_) {
                throw std::out_of_range(
                    "BackedSparseMatrixOperator::takeColumnsDense column index out of range");
            }
        }
        for (arma::uword i = 0; i < row_indices.n_elem; ++i) {
            if (row_indices(i) >= n_obs_) {
                throw std::out_of_range(
                    "BackedSparseMatrixOperator::takeColumnsDense row index out of range");
            }
        }

        arma::mat out;
        if (is_csr_) {
            take_columns_dense_csr_(col_indices, row_indices, out);
        } else {
            take_columns_dense_csc_(col_indices, row_indices, out);
        }
        return out;
    }

    arma::sp_mat BackedSparseMatrixOperator::takeColumnsSparse(
        const arma::uvec& col_indices,
        const arma::uvec& row_indices) const {

        for (arma::uword i = 0; i < col_indices.n_elem; ++i) {
            if (col_indices(i) >= n_var_) {
                throw std::out_of_range(
                    "BackedSparseMatrixOperator::takeColumnsSparse column index out of range");
            }
        }
        for (arma::uword i = 0; i < row_indices.n_elem; ++i) {
            if (row_indices(i) >= n_obs_) {
                throw std::out_of_range(
                    "BackedSparseMatrixOperator::takeColumnsSparse row index out of range");
            }
        }

        const arma::uword n_sel_cols = col_indices.n_elem;
        const bool all_rows = row_indices.is_empty();
        const arma::uword n_sel_rows = all_rows ? n_obs_ : row_indices.n_elem;

        // Build an inverse map ``src_col -> [output slot indices]`` so we
        // can emit one triplet per output slot per stored NNZ in a single
        // pass.  This replaces the previous O(D · T) post-hoc duplicate
        // expansion (D = number of duplicate slots, T = total triplets),
        // which pathologically blew up for wide gene-set queries.
        std::vector<std::vector<arma::uword>> col_to_slots(
            static_cast<size_t>(n_var_));
        for (arma::uword j = 0; j < n_sel_cols; ++j) {
            col_to_slots[static_cast<size_t>(col_indices(j))].push_back(j);
        }

        // Collect triplets then batch-construct.
        std::vector<arma::uword> trip_rows;
        std::vector<arma::uword> trip_cols;
        std::vector<double> trip_vals;

        if (is_csr_) {
            std::vector<arma::uword> row_map;
            if (!all_rows) {
                row_map.assign(static_cast<size_t>(n_obs_), n_sel_rows);
                for (arma::uword i = 0; i < n_sel_rows; ++i) {
                    if (row_map[row_indices(i)] == n_sel_rows) {
                        row_map[row_indices(i)] = i;
                    }
                }
            }

            for (arma::uword row_start = 0; row_start < n_obs_;) {
                const arma::uword row_end = std::min<arma::uword>(n_obs_, row_start + chunk_size_);
                const unsigned long long nnz_start_chunk = indptr_[row_start];
                const unsigned long long nnz_end_chunk = indptr_[row_end];
                const unsigned long long nnz_count = nnz_end_chunk - nnz_start_chunk;
                if (nnz_count == 0) {
                    row_start = row_end;
                    continue;
                }

                const std::vector<double>* data;
                const std::vector<unsigned long long>* indices;
                load_chunk_cached_(nnz_start_chunk, nnz_count, data, indices);
                ensure_chunk_transformed_csr_(row_start, row_end, nnz_start_chunk);

                for (arma::uword r = row_start; r < row_end; ++r) {
                    const arma::uword out_row = all_rows ? r : row_map[r];
                    if (out_row == n_sel_rows) continue;

                    const unsigned long long ls = indptr_[r] - nnz_start_chunk;
                    const unsigned long long le = indptr_[r + 1] - nnz_start_chunk;
                    for (unsigned long long p = ls; p < le; ++p) {
                        const arma::uword col = static_cast<arma::uword>((*indices)[static_cast<size_t>(p)]);
                        const auto& slots = col_to_slots[static_cast<size_t>(col)];
                        if (slots.empty()) continue;
                        const double v = (*data)[static_cast<size_t>(p)];
                        if (v == 0.0) continue;
                        for (arma::uword out_col : slots) {
                            trip_rows.push_back(out_row);
                            trip_cols.push_back(out_col);
                            trip_vals.push_back(v);
                        }
                    }
                }
                row_start = row_end;
            }
        } else {
            // CSC path — dedupe requested columns before the disk sweep so
            // repeated references to the same column read from HDF5 only once.
            std::vector<arma::uword> row_map;
            if (!all_rows) {
                row_map.assign(static_cast<size_t>(n_obs_), n_sel_rows);
                for (arma::uword i = 0; i < n_sel_rows; ++i) {
                    if (row_map[row_indices(i)] == n_sel_rows) {
                        row_map[row_indices(i)] = i;
                    }
                }
            }

            std::vector<arma::uword> unique_cols;
            unique_cols.reserve(n_sel_cols);
            for (arma::uword c = 0; c < n_var_; ++c) {
                if (!col_to_slots[static_cast<size_t>(c)].empty()) {
                    unique_cols.push_back(c);
                }
            }

            for (arma::uword c : unique_cols) {
                const unsigned long long nnz_start = indptr_[c];
                const unsigned long long nnz_end = indptr_[c + 1];
                const unsigned long long nnz_count = nnz_end - nnz_start;
                if (nnz_count == 0) continue;

                std::vector<double> data;
                std::vector<unsigned long long> indices;
                read_data_indices_slice_(nnz_start, nnz_count, data, indices);

                const auto& slots = col_to_slots[static_cast<size_t>(c)];
                for (unsigned long long p = 0; p < nnz_count; ++p) {
                    const arma::uword row = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const arma::uword out_row = all_rows ? row : row_map[row];
                    if (out_row == n_sel_rows) continue;
                    const double v = transform_value_(row, data[static_cast<size_t>(p)]);
                    if (v == 0.0) continue;
                    for (arma::uword out_col : slots) {
                        trip_rows.push_back(out_row);
                        trip_cols.push_back(out_col);
                        trip_vals.push_back(v);
                    }
                }
            }
        }

        if (trip_vals.empty()) {
            return arma::sp_mat(n_sel_rows, n_sel_cols);
        }

        arma::umat locations(2, trip_vals.size());
        for (size_t t = 0; t < trip_vals.size(); ++t) {
            locations(0, t) = trip_rows[t];
            locations(1, t) = trip_cols[t];
        }
        arma::vec values(trip_vals.data(), trip_vals.size(), /*copy_aux_mem=*/true);

        return arma::sp_mat(/*add_values=*/true, locations, values,
                            n_sel_rows, n_sel_cols, /*sort_locations=*/true);
    }

} // namespace actionet
