#ifndef ACTIONET_H5AD_MATRIX_IO_HPP
#define ACTIONET_H5AD_MATRIX_IO_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace actionet::h5ad {

enum class MatrixEncoding {
    Dense,
    CSR,
    CSC,
};

enum class LayoutPolicy {
    Preserve,
    Uncompressed,
};

enum class ValidationLevel {
    Structural,
    Full,
};

enum class TransformDType {
    Float32,
    Float64,
};

struct AxisSelection {
    // nullopt selects the complete axis. An explicitly present empty vector
    // selects an empty axis.
    std::optional<std::vector<std::uint64_t>> indices;

    static AxisSelection all() { return AxisSelection{}; }
    static AxisSelection from_indices(std::vector<std::uint64_t> values) {
        AxisSelection out;
        out.indices = std::move(values);
        return out;
    }
};

struct FilterInfo {
    std::uint32_t id = 0;
    std::uint32_t flags = 0;
    std::string name;
    std::vector<std::uint32_t> client_data;
    bool decode_available = false;
    bool encode_available = false;
};

struct DatasetInfo {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t logical_bytes = 0;
    std::uint64_t stored_bytes = 0;
    std::string layout;
    std::vector<std::uint64_t> chunks;
    std::vector<FilterInfo> filters;
};

struct MatrixInfo {
    MatrixEncoding encoding = MatrixEncoding::Dense;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::size_t data_item_size = 0;
    std::size_t indices_item_size = 0;
    std::size_t indptr_item_size = 0;
    bool chunked = false;
    bool filtered = false;
    std::string encoding_type;
    std::string encoding_version;
    std::uint64_t logical_bytes = 0;
    std::uint64_t stored_bytes = 0;
    std::vector<DatasetInfo> datasets;
};

struct ValidationReport {
    bool valid = false;
    MatrixInfo info;
    std::uint64_t indices_scanned = 0;
    std::string error;
};

struct SpanStats {
    std::uint64_t major_start = 0;
    std::uint64_t major_end = 0;
    std::uint64_t selected_major_entries = 0;
    std::uint64_t source_elements = 0;
    std::uint64_t source_bytes = 0;
    double source_read_seconds = 0.0;
    double packing_seconds = 0.0;
};

struct TransferOptions {
    std::size_t max_buffer_bytes = 128ULL * 1024ULL * 1024ULL;
    std::size_t gap_merge_bytes = 64ULL * 1024ULL;
    std::size_t max_rows_per_batch = 16384;
    LayoutPolicy layout_policy = LayoutPolicy::Preserve;
    bool collect_span_stats = false;
};

struct TransferStats {
    MatrixInfo source;
    MatrixInfo destination;
    std::uint64_t selected_source_bytes = 0;
    std::uint64_t source_bytes_read = 0;
    std::uint64_t gap_bytes_read = 0;
    std::uint64_t destination_bytes_written = 0;
    std::uint64_t hdf5_read_calls = 0;
    std::uint64_t hdf5_write_calls = 0;
    std::uint64_t span_count = 0;
    std::uint64_t peak_buffer_bytes = 0;
    double planning_seconds = 0.0;
    double source_read_seconds = 0.0;
    double packing_seconds = 0.0;
    double destination_write_seconds = 0.0;
    double flush_seconds = 0.0;
    double destination_fsync_seconds = 0.0;
    std::vector<SpanStats> spans;
};

struct TransformOptions {
    std::vector<double> row_scale;
    bool apply_log = false;
    double pseudocount = 1.0;
    double log_scale = 1.0;
    TransformDType output_dtype = TransformDType::Float32;
    TransferOptions transfer;
    // When present, immutable sparse indices may be hard-linked from this
    // already-existing matrix in the destination file. indptr remains
    // independent so later structural rewrites cannot alias it.
    std::optional<std::string> destination_structure_path;
    // Skip the bounded equality check only when the caller created the
    // destination structure from this exact source in the same transaction.
    bool destination_structure_is_exact_copy = false;
};

/// Inspect one H5AD dense/CSR/CSC matrix at ``group_path``.
///
/// Supported encoding versions are dense ``array`` 0.2.0 and sparse
/// ``csr_matrix``/``csc_matrix`` 0.1.0. Unsupported or malformed objects
/// throw ``std::runtime_error``.
MatrixInfo inspect_matrix(const std::string& file_path,
                          const std::string& group_path);

/// Validate a matrix. Structural validation checks shapes, sparse pointer
/// monotonicity, and payload lengths. Full validation additionally scans
/// sparse indices for bounds violations. Errors are returned in the report.
ValidationReport validate_matrix(const std::string& file_path,
                                 const std::string& group_path,
                                 ValidationLevel level = ValidationLevel::Structural);

/// Write a row/column subset from one H5AD matrix into another HDF5 file.
///
/// The destination file must already exist. Parent groups are created as
/// needed and an existing object at ``destination_group_path`` is replaced.
/// Sparse orientation and exact selector order/duplicates are preserved.
/// Sequential-scan performance guarantees apply to ordered unique selectors;
/// general selectors use bounded fallback reads.
TransferStats subset_matrix(const std::string& source_file_path,
                            const std::string& source_group_path,
                            const std::string& destination_file_path,
                            const std::string& destination_group_path,
                            const AxisSelection& row_selection,
                            const AxisSelection& column_selection,
                            const TransferOptions& options = {});

/// Copy a complete matrix, optionally changing only its storage layout.
TransferStats copy_matrix(const std::string& source_file_path,
                          const std::string& source_group_path,
                          const std::string& destination_file_path,
                          const std::string& destination_group_path,
                          const TransferOptions& options = {});

/// Apply row scaling and an optional logarithm while preserving matrix
/// orientation and sparse structure.
TransferStats transform_matrix(const std::string& source_file_path,
                               const std::string& source_group_path,
                               const std::string& destination_file_path,
                               const std::string& destination_group_path,
                               const TransformOptions& options);

} // namespace actionet::h5ad

#endif // ACTIONET_H5AD_MATRIX_IO_HPP
