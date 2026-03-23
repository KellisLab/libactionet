#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
    void check_h5(bool ok, const char* msg) {
        if (!ok) {
            throw std::runtime_error(msg);
        }
    }

    std::string normalize_encoding(const std::string& encoding) {
        std::string out = encoding;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return out;
    }
} // namespace

namespace actionet {
    std::string BackedSparseMatrixOperator::read_string_attribute_(hid_t object_id, const char* name) {
        if (H5Aexists(object_id, name) <= 0) {
            return "";
        }

        hid_t attr_id = H5Aopen(object_id, name, H5P_DEFAULT);
        check_h5(attr_id >= 0, "Failed to open string attribute");

        hid_t type_id = H5Aget_type(attr_id);
        check_h5(type_id >= 0, "Failed to get attribute type");

        std::string result;

        if (H5Tis_variable_str(type_id) > 0) {
            // Variable-length string (modern AnnData writes UTF-8 vlen strings).
            // We must set both H5T_VARIABLE size AND H5T_CSET_UTF8 on the memory
            // type, otherwise HDF5 >= 1.14 refuses to convert from a UTF-8 file
            // type to an ASCII memory type.
            hid_t mem_type = H5Tcopy(H5T_C_S1);
            H5Tset_size(mem_type, H5T_VARIABLE);
            H5Tset_cset(mem_type, H5T_CSET_UTF8);
            char* vlen_buf = nullptr;
            check_h5(H5Aread(attr_id, mem_type, &vlen_buf) >= 0,
                      "Failed to read variable-length string attribute");
            if (vlen_buf != nullptr) {
                result = vlen_buf;
                H5free_memory(vlen_buf);
            }
            H5Tclose(mem_type);
        } else {
            // Fixed-length string. H5Tget_native_type() cannot produce a
            // conversion path for UTF-8 charset strings (cset=H5T_CSET_UTF8),
            // so we read directly using the file's own type_id. Fixed-length
            // HDF5 string data is already contiguous bytes — no conversion is
            // needed; we just need the correct size.
            size_t size = H5Tget_size(type_id);
            if (size > 0) {
                std::string buffer(size, '\0');
                check_h5(H5Aread(attr_id, type_id, &buffer[0]) >= 0,
                          "Failed to read fixed-length string attribute");
                size_t null_pos = buffer.find('\0');
                if (null_pos != std::string::npos) {
                    buffer.resize(null_pos);
                }
                result = std::move(buffer);
            }
        }

        H5Tclose(type_id);
        H5Aclose(attr_id);
        return result;
    }

    std::vector<long long> BackedSparseMatrixOperator::read_shape_attribute_(hid_t object_id, const char* name) {
        hid_t attr_id = H5Aopen(object_id, name, H5P_DEFAULT);
        check_h5(attr_id >= 0, "Missing shape attribute");

        hid_t space_id = H5Aget_space(attr_id);
        check_h5(space_id >= 0, "Failed to get shape attribute dataspace");
        check_h5(H5Sget_simple_extent_ndims(space_id) == 1, "Invalid shape attribute rank");

        hsize_t dims[1] = {0};
        check_h5(H5Sget_simple_extent_dims(space_id, dims, nullptr) == 1, "Invalid shape attribute dimensions");

        std::vector<long long> shape(static_cast<size_t>(dims[0]), 0);
        check_h5(H5Aread(attr_id, H5T_NATIVE_LLONG, shape.data()) >= 0, "Failed to read shape attribute");

        H5Sclose(space_id);
        H5Aclose(attr_id);
        return shape;
    }

    BackedSparseMatrixOperator::BackedSparseMatrixOperator(
        const std::string& file_path,
        const std::string& group_path,
        arma::uword chunk_size,
        const std::vector<double>& row_scale_factors,
        bool apply_log1p)
        : file_path_(file_path),
          group_path_(group_path),
          is_csr_(true),
          apply_log1p_(apply_log1p),
          chunk_size_(std::max<arma::uword>(1, chunk_size)),
          n_obs_(0),
          n_var_(0),
          file_id_(-1),
          group_id_(-1),
          data_ds_(-1),
          indices_ds_(-1),
          indptr_ds_(-1) {

        // Disable HDF5 advisory file locking so this reader can coexist with
        // h5py/AnnData backed-mode handles that already hold a lock on the
        // same inode (errno 11 / EAGAIN from H5FD__sec2_lock otherwise).
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        check_h5(fapl >= 0, "Failed to create file access property list");
        H5Pset_file_locking(fapl, 0 /*use_file_locking=false*/, 1 /*ignore_when_disabled=true*/);
        file_id_ = H5Fopen(file_path_.c_str(), H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        check_h5(file_id_ >= 0, "Failed to open h5ad file");

        group_id_ = H5Gopen2(file_id_, group_path_.c_str(), H5P_DEFAULT);
        check_h5(group_id_ >= 0, "Failed to open sparse matrix group path");

        std::string encoding = read_string_attribute_(group_id_, "encoding-type");
        if (encoding.empty()) {
            encoding = read_string_attribute_(group_id_, "h5sparse_format");
        }
        encoding = normalize_encoding(encoding);
        if (encoding.find("csr") != std::string::npos) {
            is_csr_ = true;
        } else if (encoding.find("csc") != std::string::npos) {
            is_csr_ = false;
        } else {
            throw std::runtime_error("Unsupported sparse encoding for backed operator");
        }

        std::vector<long long> shape = read_shape_attribute_(group_id_, "shape");
        check_h5(shape.size() == 2, "Sparse shape attribute must have length 2");
        check_h5(shape[0] >= 0 && shape[1] >= 0, "Sparse shape must be non-negative");
        n_obs_ = static_cast<arma::uword>(shape[0]);
        n_var_ = static_cast<arma::uword>(shape[1]);

        data_ds_ = H5Dopen2(group_id_, "data", H5P_DEFAULT);
        indices_ds_ = H5Dopen2(group_id_, "indices", H5P_DEFAULT);
        indptr_ds_ = H5Dopen2(group_id_, "indptr", H5P_DEFAULT);
        check_h5(data_ds_ >= 0 && indices_ds_ >= 0 && indptr_ds_ >= 0,
                 "Missing sparse datasets data/indices/indptr");

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

        if (!row_scale_factors.empty()) {
            check_h5(row_scale_factors.size() == static_cast<size_t>(n_obs_),
                     "row_scale_factors length must equal n_obs");
            row_scale_ = arma::vec(row_scale_factors);
        }
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
          chunk_size_(other.chunk_size_),
          n_obs_(other.n_obs_),
          n_var_(other.n_var_),
          row_scale_(std::move(other.row_scale_)),
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
            chunk_size_ = other.chunk_size_;
            n_obs_ = other.n_obs_;
            n_var_ = other.n_var_;
            row_scale_ = std::move(other.row_scale_);
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
        std::vector<double>& data, std::vector<unsigned long long>& indices) const {
        if (chunk_cache_.count == count && chunk_cache_.start == start && count > 0) {
            data = chunk_cache_.data;
            indices = chunk_cache_.indices;
            return;
        }
        read_data_indices_slice_(start, count, data, indices);
        chunk_cache_.start = start;
        chunk_cache_.count = count;
        chunk_cache_.data = data;
        chunk_cache_.indices = indices;
    }

    double BackedSparseMatrixOperator::transform_value_(arma::uword obs_index, double value) const {
        if (!row_scale_.is_empty()) {
            value *= row_scale_(obs_index);
        }
        if (apply_log1p_) {
            value = std::log1p(value);
        }
        return value;
    }

    void BackedSparseMatrixOperator::matvec(const arma::vec& x, arma::vec& y) const {
        // S is cells × genes.  matvec: y = S * x, x is gene-length, y is cell-length.
        if (x.n_elem != n_var_) {
            throw std::runtime_error("BackedSparseMatrixOperator::matvec dimension mismatch");
        }
        if (is_csr_) {
            rmatvec_csr_(x, y);   // old rmatvec_csr_ computes sum over genes → cell-length output
        } else {
            rmatvec_csc_(x, y);
        }
    }

    void BackedSparseMatrixOperator::rmatvec(const arma::vec& x, arma::vec& y) const {
        // S is cells × genes.  rmatvec: y = S' * x, x is cell-length, y is gene-length.
        if (x.n_elem != n_obs_) {
            throw std::runtime_error("BackedSparseMatrixOperator::rmatvec dimension mismatch");
        }
        if (is_csr_) {
            matvec_csr_(x, y);    // old matvec_csr_ accumulates into gene columns → gene-length output
        } else {
            matvec_csc_(x, y);
        }
    }

    void BackedSparseMatrixOperator::matmat(const arma::mat& X, arma::mat& Y) const {
        // S is cells × genes.  matmat: Y = S * X, X is (n_var × q), Y is (n_obs × q).
        if (X.n_rows != n_var_) {
            throw std::runtime_error("BackedSparseMatrixOperator::matmat dimension mismatch");
        }
        if (is_csr_) {
            rmatmat_csr_(X, Y);   // old rmatmat_csr_ produces (n_obs × q) output
        } else {
            rmatmat_csc_(X, Y);
        }
    }

    void BackedSparseMatrixOperator::rmatmat(const arma::mat& X, arma::mat& Y) const {
        // S is cells × genes.  rmatmat: Y = S' * X, X is (n_obs × q), Y is (n_var × q).
        if (X.n_rows != n_obs_) {
            throw std::runtime_error("BackedSparseMatrixOperator::rmatmat dimension mismatch");
        }
        if (is_csr_) {
            matmat_csr_(X, Y);    // old matmat_csr_ produces (n_var × q) output
        } else {
            matmat_csc_(X, Y);
        }
    }

    void BackedSparseMatrixOperator::matvec_csr_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_var_);

        for (arma::uword row_start = 0; row_start < n_obs_; row_start += chunk_size_) {
            const arma::uword row_end = std::min<arma::uword>(n_obs_, row_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long local_start = indptr_[r] - nnz_start;
                const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                const double xval = x(r);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword col = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(r, data[static_cast<size_t>(p)]);
                    y(col) += value * xval;
                }
            }
        }
    }

    void BackedSparseMatrixOperator::rmatvec_csr_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_obs_);

        for (arma::uword row_start = 0; row_start < n_obs_; row_start += chunk_size_) {
            const arma::uword row_end = std::min<arma::uword>(n_obs_, row_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long local_start = indptr_[r] - nnz_start;
                const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                double acc = 0.0;
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword col = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(r, data[static_cast<size_t>(p)]);
                    acc += value * x(col);
                }
                y(r) = acc;
            }
        }
    }

    void BackedSparseMatrixOperator::matmat_csr_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_var_, X.n_cols);

        for (arma::uword row_start = 0; row_start < n_obs_; row_start += chunk_size_) {
            const arma::uword row_end = std::min<arma::uword>(n_obs_, row_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long local_start = indptr_[r] - nnz_start;
                const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                const arma::rowvec xrow = X.row(r);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword col = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(r, data[static_cast<size_t>(p)]);
                    Y.row(col) += value * xrow;
                }
            }
        }
    }

    void BackedSparseMatrixOperator::rmatmat_csr_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_obs_, X.n_cols);

        for (arma::uword row_start = 0; row_start < n_obs_; row_start += chunk_size_) {
            const arma::uword row_end = std::min<arma::uword>(n_obs_, row_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const unsigned long long local_start = indptr_[r] - nnz_start;
                const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                arma::rowvec acc(X.n_cols, arma::fill::zeros);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword col = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(r, data[static_cast<size_t>(p)]);
                    acc += value * X.row(col);
                }
                Y.row(r) = acc;
            }
        }
    }

    void BackedSparseMatrixOperator::matvec_csc_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_var_);

        for (arma::uword col_start = 0; col_start < n_var_; col_start += chunk_size_) {
            const arma::uword col_end = std::min<arma::uword>(n_var_, col_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long local_start = indptr_[c] - nnz_start;
                const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                double acc = 0.0;
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword row = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(row, data[static_cast<size_t>(p)]);
                    acc += value * x(row);
                }
                y(c) = acc;
            }
        }
    }

    void BackedSparseMatrixOperator::rmatvec_csc_(const arma::vec& x, arma::vec& y) const {
        y.zeros(n_obs_);

        for (arma::uword col_start = 0; col_start < n_var_; col_start += chunk_size_) {
            const arma::uword col_end = std::min<arma::uword>(n_var_, col_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long local_start = indptr_[c] - nnz_start;
                const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                const double xval = x(c);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword row = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(row, data[static_cast<size_t>(p)]);
                    y(row) += value * xval;
                }
            }
        }
    }

    void BackedSparseMatrixOperator::matmat_csc_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_var_, X.n_cols);

        for (arma::uword col_start = 0; col_start < n_var_; col_start += chunk_size_) {
            const arma::uword col_end = std::min<arma::uword>(n_var_, col_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long local_start = indptr_[c] - nnz_start;
                const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                arma::rowvec acc(X.n_cols, arma::fill::zeros);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword row = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(row, data[static_cast<size_t>(p)]);
                    acc += value * X.row(row);
                }
                Y.row(c) = acc;
            }
        }
    }

    void BackedSparseMatrixOperator::rmatmat_csc_(const arma::mat& X, arma::mat& Y) const {
        Y.zeros(n_obs_, X.n_cols);

        for (arma::uword col_start = 0; col_start < n_var_; col_start += chunk_size_) {
            const arma::uword col_end = std::min<arma::uword>(n_var_, col_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[col_start];
            const unsigned long long nnz_end = indptr_[col_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword c = col_start; c < col_end; ++c) {
                const unsigned long long local_start = indptr_[c] - nnz_start;
                const unsigned long long local_end = indptr_[c + 1] - nnz_start;
                const arma::rowvec xrow = X.row(c);
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword row = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const double value = transform_value_(row, data[static_cast<size_t>(p)]);
                    Y.row(row) += value * xrow;
                }
            }
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

        // Build col -> output-position lookup (sentinel = n_sel_cols means "skip").
        std::vector<arma::uword> col_map(static_cast<size_t>(n_var_), n_sel_cols);
        for (arma::uword j = 0; j < n_sel_cols; ++j) {
            const arma::uword c = col_indices(j);
            if (col_map[c] == n_sel_cols) {
                col_map[c] = j;
            }
        }

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

        for (arma::uword row_start = 0; row_start < n_obs_; row_start += chunk_size_) {
            const arma::uword row_end = std::min<arma::uword>(n_obs_, row_start + chunk_size_);
            const unsigned long long nnz_start = indptr_[row_start];
            const unsigned long long nnz_end = indptr_[row_end];
            const unsigned long long nnz_count = nnz_end - nnz_start;
            if (nnz_count == 0) continue;

            std::vector<double> data;
            std::vector<unsigned long long> indices;
            load_chunk_cached_(nnz_start, nnz_count, data, indices);

            for (arma::uword r = row_start; r < row_end; ++r) {
                const arma::uword out_row = all_rows ? r : row_map[r];
                if (out_row == n_sel_rows) continue;

                const unsigned long long local_start = indptr_[r] - nnz_start;
                const unsigned long long local_end = indptr_[r + 1] - nnz_start;
                for (unsigned long long p = local_start; p < local_end; ++p) {
                    const arma::uword col = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                    const arma::uword out_col = col_map[col];
                    if (out_col == n_sel_cols) continue;
                    out(out_row, out_col) = transform_value_(r, data[static_cast<size_t>(p)]);
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
    }

    arma::mat BackedSparseMatrixOperator::takeColumnsDense(
        const arma::uvec& col_indices,
        const arma::uvec& row_indices) const {

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

        const arma::uword n_sel_cols = col_indices.n_elem;
        const bool all_rows = row_indices.is_empty();
        const arma::uword n_sel_rows = all_rows ? n_obs_ : row_indices.n_elem;

        // Collect triplets then batch-construct.
        std::vector<arma::uword> trip_rows;
        std::vector<arma::uword> trip_cols;
        std::vector<double> trip_vals;

        if (is_csr_) {
            // col -> output-position lookup
            std::vector<arma::uword> col_map(static_cast<size_t>(n_var_), n_sel_cols);
            for (arma::uword j = 0; j < n_sel_cols; ++j) {
                if (col_map[col_indices(j)] == n_sel_cols) {
                    col_map[col_indices(j)] = j;
                }
            }

            std::vector<arma::uword> row_map;
            if (!all_rows) {
                row_map.assign(static_cast<size_t>(n_obs_), n_sel_rows);
                for (arma::uword i = 0; i < n_sel_rows; ++i) {
                    if (row_map[row_indices(i)] == n_sel_rows) {
                        row_map[row_indices(i)] = i;
                    }
                }
            }

            for (arma::uword row_start = 0; row_start < n_obs_; row_start += chunk_size_) {
                const arma::uword row_end = std::min<arma::uword>(n_obs_, row_start + chunk_size_);
                const unsigned long long nnz_start_chunk = indptr_[row_start];
                const unsigned long long nnz_end_chunk = indptr_[row_end];
                const unsigned long long nnz_count = nnz_end_chunk - nnz_start_chunk;
                if (nnz_count == 0) continue;

                std::vector<double> data;
                std::vector<unsigned long long> indices;
                load_chunk_cached_(nnz_start_chunk, nnz_count, data, indices);

                for (arma::uword r = row_start; r < row_end; ++r) {
                    const arma::uword out_row = all_rows ? r : row_map[r];
                    if (out_row == n_sel_rows) continue;

                    const unsigned long long ls = indptr_[r] - nnz_start_chunk;
                    const unsigned long long le = indptr_[r + 1] - nnz_start_chunk;
                    for (unsigned long long p = ls; p < le; ++p) {
                        const arma::uword col = static_cast<arma::uword>(indices[static_cast<size_t>(p)]);
                        const arma::uword out_col = col_map[col];
                        if (out_col == n_sel_cols) continue;
                        double v = transform_value_(r, data[static_cast<size_t>(p)]);
                        if (v != 0.0) {
                            trip_rows.push_back(out_row);
                            trip_cols.push_back(out_col);
                            trip_vals.push_back(v);
                        }
                    }
                }
            }

            // Expand duplicate columns.
            for (arma::uword j = 0; j < n_sel_cols; ++j) {
                if (col_map[col_indices(j)] != j) {
                    const arma::uword src_j = col_map[col_indices(j)];
                    const size_t n = trip_rows.size();
                    for (size_t t = 0; t < n; ++t) {
                        if (trip_cols[t] == src_j) {
                            trip_rows.push_back(trip_rows[t]);
                            trip_cols.push_back(j);
                            trip_vals.push_back(trip_vals[t]);
                        }
                    }
                }
            }
        } else {
            // CSC path
            std::vector<arma::uword> row_map;
            if (!all_rows) {
                row_map.assign(static_cast<size_t>(n_obs_), n_sel_rows);
                for (arma::uword i = 0; i < n_sel_rows; ++i) {
                    if (row_map[row_indices(i)] == n_sel_rows) {
                        row_map[row_indices(i)] = i;
                    }
                }
            }

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
                    double v = transform_value_(row, data[static_cast<size_t>(p)]);
                    if (v != 0.0) {
                        trip_rows.push_back(out_row);
                        trip_cols.push_back(j);
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
