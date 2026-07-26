#include "io/backed_h5ad/h5ad_matrix_io.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_string_attr(hid_t object, const char* name, const char* value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, H5T_VARIABLE);
    H5Tset_cset(type, H5T_CSET_UTF8);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(
        object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    require(attr >= 0, "failed to create string attribute");
    require(H5Awrite(attr, type, &value) >= 0, "failed to write string attribute");
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
}

void write_shape_attr(hid_t group, std::uint64_t rows, std::uint64_t cols) {
    const long long values[2] = {
        static_cast<long long>(rows), static_cast<long long>(cols)};
    const hsize_t dims[1] = {2};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t attr = H5Acreate2(
        group, "shape", H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT);
    require(attr >= 0, "failed to create shape attribute");
    require(H5Awrite(attr, H5T_NATIVE_LLONG, values) >= 0,
            "failed to write shape attribute");
    H5Aclose(attr);
    H5Sclose(space);
}

hid_t create_dcpl(const std::vector<hsize_t>& chunks, bool gzip) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    require(dcpl >= 0, "failed to create dcpl");
    if (!chunks.empty()) {
        require(H5Pset_chunk(
                    dcpl, static_cast<int>(chunks.size()), chunks.data()) >= 0,
                "failed to set chunks");
    }
    if (gzip) {
        require(H5Pset_shuffle(dcpl) >= 0, "failed to set shuffle");
        require(H5Pset_deflate(dcpl, 1) >= 0, "failed to set gzip");
        require(H5Pset_fletcher32(dcpl) >= 0, "failed to set checksum");
    }
    return dcpl;
}

template<class T>
hid_t create_1d(hid_t parent, const char* name, hid_t file_type, hid_t mem_type,
                const std::vector<T>& values, bool gzip) {
    const hsize_t dims[1] = {values.size()};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dcpl = create_dcpl(
        {std::max<hsize_t>(1, std::min<hsize_t>(4, dims[0]))}, gzip);
    hid_t dataset = H5Dcreate2(
        parent, name, file_type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    require(dataset >= 0, "failed to create 1d dataset");
    if (!values.empty()) {
        require(H5Dwrite(
                    dataset, mem_type, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, values.data()) >= 0,
                "failed to write 1d dataset");
    }
    H5Pclose(dcpl);
    H5Sclose(space);
    return dataset;
}

void create_dense(hid_t file, const std::vector<long long>& dense,
                  std::uint64_t rows, std::uint64_t cols) {
    const hsize_t dims[2] = {rows, cols};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dcpl = create_dcpl({2, 2}, true);
    hid_t dataset = H5Dcreate2(
        file, "/dense", H5T_STD_I64LE, space,
        H5P_DEFAULT, dcpl, H5P_DEFAULT);
    require(dataset >= 0, "failed to create dense matrix");
    require(H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, dense.data()) >= 0,
            "failed to write dense matrix");
    write_string_attr(dataset, "encoding-type", "array");
    write_string_attr(dataset, "encoding-version", "0.2.0");
    H5Dclose(dataset);
    H5Pclose(dcpl);
    H5Sclose(space);
}

void create_contiguous_dense(
    hid_t file,
    const char* path,
    const std::vector<long long>& dense,
    std::uint64_t rows,
    std::uint64_t cols,
    bool unavailable_filter = false) {
    const hsize_t dims[2] = {rows, cols};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dcpl = H5P_DEFAULT;
    hid_t owned_dcpl = -1;
    if (unavailable_filter) {
        owned_dcpl = create_dcpl({2, 2}, false);
        require(H5Pset_filter(
                    owned_dcpl, static_cast<H5Z_filter_t>(32001),
                    H5Z_FLAG_OPTIONAL, 0, nullptr) >= 0,
                "failed to set unavailable optional filter");
        dcpl = owned_dcpl;
    }
    hid_t dataset = H5Dcreate2(
        file, path, H5T_STD_I64LE, space,
        H5P_DEFAULT, dcpl, H5P_DEFAULT);
    require(dataset >= 0, "failed to create dense fixture");
    require(H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, dense.data()) >= 0,
            "failed to write dense fixture");
    write_string_attr(dataset, "encoding-type", "array");
    write_string_attr(dataset, "encoding-version", "0.2.0");
    H5Dclose(dataset);
    if (owned_dcpl >= 0) {
        H5Pclose(owned_dcpl);
    }
    H5Sclose(space);
}

void create_sparse(
    hid_t file,
    const char* path,
    bool csr,
    const std::vector<long long>& data,
    const std::vector<unsigned int>& indices,
    const std::vector<unsigned long long>& indptr,
    std::uint64_t rows,
    std::uint64_t cols) {
    hid_t group = H5Gcreate2(
        file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    require(group >= 0, "failed to create sparse group");
    write_string_attr(group, "encoding-type", csr ? "csr_matrix" : "csc_matrix");
    write_string_attr(group, "encoding-version", "0.1.0");
    write_shape_attr(group, rows, cols);
    H5Dclose(create_1d(
        group, "data", H5T_STD_I64LE, H5T_NATIVE_LLONG, data, true));
    H5Dclose(create_1d(
        group, "indices", H5T_STD_U32LE, H5T_NATIVE_UINT, indices, true));
    H5Dclose(create_1d(
        group, "indptr", H5T_STD_U64LE, H5T_NATIVE_ULLONG, indptr, true));
    H5Gclose(group);
}

void create_signed_sparse(
    hid_t file,
    const char* path,
    const std::vector<long long>& data,
    const std::vector<long long>& indices,
    const std::vector<long long>& indptr,
    std::uint64_t rows,
    std::uint64_t cols) {
    hid_t group = H5Gcreate2(
        file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_string_attr(group, "encoding-type", "csr_matrix");
    write_string_attr(group, "encoding-version", "0.1.0");
    write_shape_attr(group, rows, cols);
    H5Dclose(create_1d(
        group, "data", H5T_STD_I64LE, H5T_NATIVE_LLONG, data, false));
    H5Dclose(create_1d(
        group, "indices", H5T_STD_I64LE, H5T_NATIVE_LLONG, indices, false));
    H5Dclose(create_1d(
        group, "indptr", H5T_STD_I64LE, H5T_NATIVE_LLONG, indptr, false));
    H5Gclose(group);
}

std::vector<long long> read_dense(hid_t file, const std::string& path) {
    hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    require(dataset >= 0, "failed to open dense result");
    hid_t space = H5Dget_space(dataset);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    std::vector<long long> values(
        static_cast<std::size_t>(dims[0] * dims[1]), 0);
    if (!values.empty()) {
        require(H5Dread(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, values.data()) >= 0,
                "failed to read dense result");
    }
    H5Sclose(space);
    H5Dclose(dataset);
    return values;
}

std::vector<unsigned long long> read_u64_1d(hid_t parent, const char* name) {
    hid_t dataset = H5Dopen2(parent, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dataset);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    std::vector<unsigned long long> values(dims[0], 0);
    if (!values.empty()) {
        require(H5Dread(dataset, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, values.data()) >= 0,
                "failed to read integer result");
    }
    H5Sclose(space);
    H5Dclose(dataset);
    return values;
}

std::vector<long long> sparse_to_dense(
    hid_t file, const std::string& path, bool csr,
    std::uint64_t rows, std::uint64_t cols) {
    hid_t group = H5Gopen2(file, path.c_str(), H5P_DEFAULT);
    const auto data_unsigned = read_u64_1d(group, "data");
    const auto indices = read_u64_1d(group, "indices");
    const auto indptr = read_u64_1d(group, "indptr");
    std::vector<long long> out(rows * cols, 0);
    for (std::size_t major = 0; major + 1 < indptr.size(); ++major) {
        for (auto position = indptr[major]; position < indptr[major + 1]; ++position) {
            const auto minor = indices[position];
            const auto row = csr ? major : minor;
            const auto col = csr ? minor : major;
            out[row * cols + col] = static_cast<long long>(data_unsigned[position]);
        }
    }
    H5Gclose(group);
    return out;
}

std::vector<long long> dense_reference(
    const std::vector<long long>& source,
    std::uint64_t source_cols,
    const std::vector<std::uint64_t>& rows,
    const std::vector<std::uint64_t>& cols) {
    std::vector<long long> out;
    for (const auto row : rows) {
        for (const auto col : cols) {
            out.push_back(source[row * source_cols + col]);
        }
    }
    return out;
}

} // namespace

int main() {
    const auto base = std::filesystem::temp_directory_path();
    const auto source_path = (base / "actionet_h5ad_native_source.h5").string();
    const auto destination_path = (base / "actionet_h5ad_native_destination.h5").string();
    std::filesystem::remove(source_path);
    std::filesystem::remove(destination_path);

    const std::vector<long long> dense = {
        0, 9007199254740995LL, 0, 4,
        5, 0, 6, 0,
        0, 7, 0, 8,
        9, 0, 10, 11,
        12, 13, 0, 0,
    };
    const std::vector<long long> csr_data = {
        9007199254740995LL, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const std::vector<unsigned int> csr_indices = {
        1, 3, 0, 2, 1, 3, 0, 2, 3, 0, 1};
    const std::vector<unsigned long long> csr_indptr = {0, 2, 4, 6, 9, 11};
    const std::vector<long long> csc_data = {
        5, 9, 12, 9007199254740995LL, 7, 13, 6, 10, 4, 8, 11};
    const std::vector<unsigned int> csc_indices = {
        1, 3, 4, 0, 2, 4, 1, 3, 0, 2, 3};
    const std::vector<unsigned long long> csc_indptr = {0, 3, 6, 8, 11};

    hid_t source = H5Fcreate(
        source_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    require(source >= 0, "failed to create source file");
    create_dense(source, dense, 5, 4);
    create_contiguous_dense(source, "/dense_contiguous", dense, 5, 4);
    create_contiguous_dense(
        source, "/unavailable_filter", dense, 5, 4, true);
    create_sparse(
        source, "/csr", true, csr_data, csr_indices, csr_indptr, 5, 4);
    create_sparse(
        source, "/csc", false, csc_data, csc_indices, csc_indptr, 5, 4);
    create_sparse(
        source, "/empty", true,
        std::vector<long long>{},
        std::vector<unsigned int>{},
        std::vector<unsigned long long>{0, 0, 0},
        2, 3);
    create_sparse(
        source, "/bad_indptr", true,
        std::vector<long long>{1},
        std::vector<unsigned int>{0},
        std::vector<unsigned long long>{0, 2},
        1, 2);
    create_sparse(
        source, "/bad_index", true,
        std::vector<long long>{1},
        std::vector<unsigned int>{5},
        std::vector<unsigned long long>{0, 1},
        1, 2);
    create_signed_sparse(
        source, "/signed_indices", csr_data,
        std::vector<long long>(csr_indices.begin(), csr_indices.end()),
        std::vector<long long>(csr_indptr.begin(), csr_indptr.end()),
        5, 4);
    require(H5Ocopy(
                source, "/dense_contiguous", source, "/unknown_encoding",
                H5P_DEFAULT, H5P_DEFAULT) >= 0,
            "failed to copy unknown-encoding fixture");
    hid_t unknown = H5Dopen2(source, "/unknown_encoding", H5P_DEFAULT);
    require(H5Adelete(unknown, "encoding-version") >= 0,
            "failed to replace encoding version");
    write_string_attr(unknown, "encoding-version", "9.9.9");
    H5Dclose(unknown);
    H5Fclose(source);
    hid_t destination = H5Fcreate(
        destination_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(destination);

    const auto info = actionet::h5ad::inspect_matrix(source_path, "/csr");
    require(info.rows == 5 && info.cols == 4 && info.nnz == 11,
            "sparse inspection mismatch");
    require(info.filtered, "gzip fixture was not detected as filtered");
    require(info.datasets.size() == 3 &&
                info.datasets[0].name == "data" &&
                info.datasets[0].dtype == "int64" &&
                info.datasets[0].layout == "chunked" &&
                info.datasets[0].chunks == std::vector<std::uint64_t>{4} &&
                info.datasets[0].filters.size() == 3 &&
                info.logical_bytes > 0 && info.stored_bytes > 0,
            "sparse storage inventory mismatch");
    require(actionet::h5ad::validate_matrix(
                source_path, "/csr", actionet::h5ad::ValidationLevel::Full).valid,
            "valid sparse fixture was rejected");
    require(actionet::h5ad::inspect_matrix(source_path, "/empty").nnz == 0,
            "empty sparse fixture inspection failed");
    require(!actionet::h5ad::validate_matrix(
                 source_path, "/bad_indptr",
                 actionet::h5ad::ValidationLevel::Structural).valid,
            "malformed sparse indptr was accepted");
    require(actionet::h5ad::validate_matrix(
                source_path, "/bad_index",
                actionet::h5ad::ValidationLevel::Structural).valid &&
                !actionet::h5ad::validate_matrix(
                     source_path, "/bad_index",
                     actionet::h5ad::ValidationLevel::Full).valid,
            "full validation did not reject an invalid sparse index");
    bool unknown_encoding_rejected = false;
    try {
        actionet::h5ad::inspect_matrix(source_path, "/unknown_encoding");
    } catch (const std::exception&) {
        unknown_encoding_rejected = true;
    }
    require(unknown_encoding_rejected, "unknown encoding version was accepted");

    const std::vector<std::vector<std::uint64_t>> row_cases = {
        {0, 1, 2, 3, 4}, {}, {2}, {1, 2, 3}, {0, 2, 4}, {4, 0, 4, 1},
        {4, 3, 2, 1, 0}};
    const std::vector<std::vector<std::uint64_t>> col_cases = {
        {0, 1, 2, 3}, {}, {3}, {1, 2}, {3, 1, 3, 0}, {3, 2, 1, 0}};
    actionet::h5ad::TransferOptions options;
    options.max_buffer_bytes = 1024;
    options.max_rows_per_batch = 2;
    options.collect_span_stats = true;

    actionet::h5ad::copy_matrix(
        source_path, "/signed_indices",
        destination_path, "/signed_indices_copy", options);
    require(actionet::h5ad::validate_matrix(
                destination_path, "/signed_indices_copy",
                actionet::h5ad::ValidationLevel::Full).valid,
            "signed sparse indices did not transfer");
    actionet::h5ad::copy_matrix(
        source_path, "/dense_contiguous",
        destination_path, "/dense_contiguous_copy", options);
    require(actionet::h5ad::inspect_matrix(
                destination_path, "/dense_contiguous_copy")
                .datasets[0].layout == "contiguous",
            "contiguous dense layout was not preserved");

    hid_t sentinel_file = H5Fopen(
        destination_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    const hsize_t sentinel_dims[1] = {1};
    hid_t sentinel_space = H5Screate_simple(1, sentinel_dims, nullptr);
    hid_t sentinel = H5Dcreate2(
        sentinel_file, "/sentinel", H5T_STD_I64LE, sentinel_space,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const long long sentinel_value = 42;
    H5Dwrite(
        sentinel, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, &sentinel_value);
    H5Dclose(sentinel);
    H5Sclose(sentinel_space);
    H5Fclose(sentinel_file);
    bool unavailable_filter_rejected = false;
    try {
        actionet::h5ad::copy_matrix(
            source_path, "/unavailable_filter",
            destination_path, "/sentinel", options);
    } catch (const std::exception&) {
        unavailable_filter_rejected = true;
    }
    require(
        unavailable_filter_rejected,
        "unavailable filter encoder was accepted");
    sentinel_file = H5Fopen(
        destination_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    sentinel = H5Dopen2(sentinel_file, "/sentinel", H5P_DEFAULT);
    long long observed_sentinel = 0;
    H5Dread(
        sentinel, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, &observed_sentinel);
    H5Dclose(sentinel);
    H5Fclose(sentinel_file);
    require(
        observed_sentinel == sentinel_value,
        "capability rejection mutated the destination");

    std::size_t case_number = 0;
    for (const auto& rows : row_cases) {
        for (const auto& cols : col_cases) {
            const auto expected = dense_reference(dense, 4, rows, cols);
            for (const std::string encoding : {"dense", "csr", "csc"}) {
                const std::string output =
                    "/case_" + std::to_string(case_number++) + "_" + encoding;
                const auto stats = actionet::h5ad::subset_matrix(
                    source_path, "/" + encoding, destination_path, output,
                    actionet::h5ad::AxisSelection::from_indices(rows),
                    actionet::h5ad::AxisSelection::from_indices(cols),
                    options);
                require(stats.destination.rows == rows.size() &&
                            stats.destination.cols == cols.size(),
                        "destination shape mismatch");
                require(!stats.destination.datasets.empty(),
                        "destination storage inventory is empty");
                hid_t file = H5Fopen(
                    destination_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                const auto observed = encoding == "dense"
                    ? read_dense(file, output)
                    : sparse_to_dense(
                          file, output, encoding == "csr",
                          rows.size(), cols.size());
                H5Fclose(file);
                require(observed == expected,
                        "native selector result differs from dense reference");
            }
        }
    }

    std::mt19937_64 rng(1729);
    for (std::size_t trial = 0; trial < 40; ++trial) {
        std::vector<std::uint64_t> rows(rng() % 8);
        std::vector<std::uint64_t> cols(rng() % 7);
        for (auto& row : rows) {
            row = rng() % 5;
        }
        for (auto& col : cols) {
            col = rng() % 4;
        }
        const auto expected = dense_reference(dense, 4, rows, cols);
        for (const std::string encoding : {"dense", "csr", "csc"}) {
            const std::string output =
                "/property_" + std::to_string(trial) + "_" + encoding;
            actionet::h5ad::subset_matrix(
                source_path, "/" + encoding, destination_path, output,
                actionet::h5ad::AxisSelection::from_indices(rows),
                actionet::h5ad::AxisSelection::from_indices(cols),
                options);
            hid_t file = H5Fopen(
                destination_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            const auto observed = encoding == "dense"
                ? read_dense(file, output)
                : sparse_to_dense(
                      file, output, encoding == "csr",
                      rows.size(), cols.size());
            H5Fclose(file);
            require(observed == expected,
                    "random native selector differs from dense reference");
        }
    }

    actionet::h5ad::TransferOptions uncompressed = options;
    uncompressed.layout_policy = actionet::h5ad::LayoutPolicy::Uncompressed;
    actionet::h5ad::copy_matrix(
        source_path, "/csr", destination_path, "/uncompressed", uncompressed);
    const auto copied = actionet::h5ad::inspect_matrix(
        destination_path, "/uncompressed");
    require(!copied.filtered, "Uncompressed policy retained an HDF5 filter");

    actionet::h5ad::copy_matrix(
        source_path, "/csr", destination_path, "/structure", options);
    actionet::h5ad::TransformOptions transform;
    transform.row_scale = {0.0, 0.5, 1.0, 3.0, 0.25};
    transform.apply_log = false;
    transform.output_dtype = actionet::h5ad::TransformDType::Float64;
    transform.transfer = options;
    transform.destination_structure_path = "/structure";
    actionet::h5ad::transform_matrix(
        source_path, "/csr", destination_path, "/transformed", transform);
    hid_t transformed_file = H5Fopen(
        destination_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const auto transformed = sparse_to_dense(
        transformed_file, "/transformed", true, 5, 4);
    H5Fclose(transformed_file);
    const std::vector<long long> transformed_expected = {
        0, 0, 0, 0,
        2, 0, 3, 0,
        0, 7, 0, 8,
        27, 0, 30, 33,
        3, 3, 0, 0,
    };
    require(transformed == transformed_expected,
            "native sparse transform differs from expected values");

    std::filesystem::remove(source_path);
    std::filesystem::remove(destination_path);
    std::cout << "h5ad native matrix I/O tests passed\n";
    return 0;
}
