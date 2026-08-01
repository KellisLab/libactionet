#include "io/backed_h5ad/h5ad_matrix_io.hpp"

#include "_h5_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

using actionet::detail::h5::check_h5;
using actionet::detail::h5::flush_and_fsync_file;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

// Drain accumulated dirty output pages to disk roughly every this many bytes
// written, instead of leaving the whole multi-GB output dirty for a single
// trailing fsync. 512 MiB overlaps writeback with ongoing packing/reads while
// keeping the periodic-flush count small on atlas-scale transfers.
constexpr std::uint64_t kIncrementalFsyncBytes = 512ULL * 1024ULL * 1024ULL;

using H5File = actionet::detail::h5::File;
using H5Group = actionet::detail::h5::Group;
using H5Dataset = actionet::detail::h5::Dataset;
using H5Space = actionet::detail::h5::Space;
using H5Type = actionet::detail::h5::Type;
using H5Attr = actionet::detail::h5::Attribute;
using H5Property = actionet::detail::h5::Property;

std::string normalize_h5_path(const std::string& path) {
    if (path.empty() || path == "/") {
        throw std::runtime_error("H5AD matrix path must not be empty or '/'");
    }
    return path.front() == '/' ? path : "/" + path;
}

std::pair<std::string, std::string> split_parent_name(const std::string& raw_path) {
    const std::string path = normalize_h5_path(raw_path);
    const auto slash = path.find_last_of('/');
    const std::string parent = slash == 0 ? "/" : path.substr(0, slash);
    const std::string name = path.substr(slash + 1);
    if (name.empty()) {
        throw std::runtime_error("Invalid HDF5 object path: " + raw_path);
    }
    return {parent, name};
}

H5File open_readonly(const std::string& path, const char* context) {
    return H5File(actionet::detail::h5::open_h5_readonly_no_lock(path, context));
}

H5File open_readwrite(const std::string& path, const char* context) {
    H5Property fapl(H5Pcreate(H5P_FILE_ACCESS));
    check_h5(static_cast<bool>(fapl), "Failed to create HDF5 file access property list");
    H5Pset_file_locking(fapl.get(), 0, 1);
    const hid_t id = H5Fopen(path.c_str(), H5F_ACC_RDWR, fapl.get());
    if (id < 0) {
        throw std::runtime_error(std::string(context) +
                                 ": failed to open destination HDF5 file: " + path);
    }
    return H5File(id);
}

std::string read_string_attr(hid_t object, const char* name) {
    if (H5Aexists(object, name) <= 0) {
        return "";
    }
    H5Attr attr(H5Aopen(object, name, H5P_DEFAULT));
    check_h5(static_cast<bool>(attr), "Failed to open HDF5 string attribute");
    H5Type type(H5Aget_type(attr.get()));
    check_h5(static_cast<bool>(type), "Failed to inspect HDF5 string attribute");

    if (H5Tget_class(type.get()) != H5T_STRING) {
        throw std::runtime_error(std::string("Attribute '") + name + "' is not a string");
    }

    if (H5Tis_variable_str(type.get()) > 0) {
        H5Type mem(H5Tcopy(H5T_C_S1));
        check_h5(static_cast<bool>(mem), "Failed to copy HDF5 string type");
        check_h5(H5Tset_size(mem.get(), H5T_VARIABLE) >= 0,
                 "Failed to set variable HDF5 string size");
        check_h5(H5Tset_cset(mem.get(), H5T_CSET_UTF8) >= 0,
                 "Failed to set UTF-8 HDF5 string type");
        char* value = nullptr;
        check_h5(H5Aread(attr.get(), mem.get(), &value) >= 0,
                 "Failed to read variable HDF5 string attribute");
        std::string out = value == nullptr ? "" : value;
        if (value != nullptr) {
            H5free_memory(value);
        }
        return out;
    }

    const std::size_t size = H5Tget_size(type.get());
    std::string out(size, '\0');
    if (size > 0) {
        check_h5(H5Aread(attr.get(), type.get(), out.data()) >= 0,
                 "Failed to read fixed HDF5 string attribute");
        const auto nul = out.find('\0');
        if (nul != std::string::npos) {
            out.resize(nul);
        }
    }
    return out;
}

void write_string_attr(hid_t object, const char* name, const std::string& value) {
    H5Type type(H5Tcopy(H5T_C_S1));
    check_h5(static_cast<bool>(type), "Failed to create HDF5 string attribute type");
    check_h5(H5Tset_size(type.get(), H5T_VARIABLE) >= 0,
             "Failed to set HDF5 variable string size");
    check_h5(H5Tset_cset(type.get(), H5T_CSET_UTF8) >= 0,
             "Failed to set HDF5 string charset");
    H5Space scalar(H5Screate(H5S_SCALAR));
    check_h5(static_cast<bool>(scalar), "Failed to create scalar HDF5 dataspace");
    H5Attr attr(H5Acreate2(object, name, type.get(), scalar.get(), H5P_DEFAULT, H5P_DEFAULT));
    check_h5(static_cast<bool>(attr), "Failed to create HDF5 string attribute");
    const char* ptr = value.c_str();
    check_h5(H5Awrite(attr.get(), type.get(), &ptr) >= 0,
             "Failed to write HDF5 string attribute");
}

std::vector<std::uint64_t> read_shape_attr(hid_t object) {
    if (H5Aexists(object, "shape") <= 0) {
        throw std::runtime_error("Missing H5AD sparse shape attribute");
    }
    H5Attr attr(H5Aopen(object, "shape", H5P_DEFAULT));
    H5Space space(H5Aget_space(attr.get()));
    check_h5(static_cast<bool>(attr) && static_cast<bool>(space),
             "Failed to open H5AD sparse shape attribute");
    check_h5(H5Sget_simple_extent_ndims(space.get()) == 1,
             "H5AD sparse shape attribute must be one-dimensional");
    hsize_t dims[1] = {0};
    check_h5(H5Sget_simple_extent_dims(space.get(), dims, nullptr) == 1 && dims[0] == 2,
             "H5AD sparse shape attribute must have length two");
    std::vector<long long> signed_shape(2, 0);
    check_h5(H5Aread(attr.get(), H5T_NATIVE_LLONG, signed_shape.data()) >= 0,
             "Failed to read H5AD sparse shape");
    if (signed_shape[0] < 0 || signed_shape[1] < 0) {
        throw std::runtime_error("H5AD sparse shape contains a negative dimension");
    }
    return {static_cast<std::uint64_t>(signed_shape[0]),
            static_cast<std::uint64_t>(signed_shape[1])};
}

void write_shape_attr(hid_t object, std::uint64_t rows, std::uint64_t cols) {
    if (rows > static_cast<std::uint64_t>(std::numeric_limits<long long>::max()) ||
        cols > static_cast<std::uint64_t>(std::numeric_limits<long long>::max())) {
        throw std::runtime_error("H5AD matrix shape exceeds signed 64-bit storage");
    }
    const long long shape[2] = {static_cast<long long>(rows), static_cast<long long>(cols)};
    const hsize_t dims[1] = {2};
    H5Space space(H5Screate_simple(1, dims, nullptr));
    H5Attr attr(H5Acreate2(object, "shape", H5T_STD_I64LE, space.get(),
                           H5P_DEFAULT, H5P_DEFAULT));
    check_h5(static_cast<bool>(space) && static_cast<bool>(attr),
             "Failed to create H5AD sparse shape attribute");
    check_h5(H5Awrite(attr.get(), H5T_NATIVE_LLONG, shape) >= 0,
             "Failed to write H5AD sparse shape attribute");
}

std::vector<hsize_t> dataset_shape(hid_t dataset) {
    H5Space space(H5Dget_space(dataset));
    check_h5(static_cast<bool>(space), "Failed to open HDF5 dataset dataspace");
    const int rank = H5Sget_simple_extent_ndims(space.get());
    check_h5(rank >= 0, "Failed to inspect HDF5 dataset rank");
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank), 0);
    if (rank > 0) {
        check_h5(H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr) == rank,
                 "Failed to inspect HDF5 dataset dimensions");
    }
    return dims;
}

void require_numeric_type(hid_t dataset, const char* name) {
    H5Type type(H5Dget_type(dataset));
    check_h5(static_cast<bool>(type), "Failed to inspect HDF5 dataset type");
    const H5T_class_t cls = H5Tget_class(type.get());
    if (cls != H5T_INTEGER && cls != H5T_FLOAT) {
        throw std::runtime_error(std::string("H5AD matrix dataset '") + name +
                                 "' must have an integer or floating dtype");
    }
}

void require_integer_type(hid_t dataset, const char* name) {
    H5Type type(H5Dget_type(dataset));
    check_h5(static_cast<bool>(type), "Failed to inspect HDF5 dataset type");
    if (H5Tget_class(type.get()) != H5T_INTEGER) {
        throw std::runtime_error(std::string("H5AD sparse dataset '") + name +
                                 "' must have an integer dtype");
    }
}

std::uint64_t checked_product(hsize_t a, hsize_t b, const char* context) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::runtime_error(std::string(context) + " exceeds uint64 range");
    }
    return static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
}

std::vector<std::uint64_t> read_integer_1d(hid_t dataset) {
    const auto dims = dataset_shape(dataset);
    check_h5(dims.size() == 1, "H5AD sparse index datasets must be one-dimensional");
    const std::size_t count = static_cast<std::size_t>(dims[0]);
    H5Type type(H5Dget_type(dataset));
    check_h5(static_cast<bool>(type) && H5Tget_class(type.get()) == H5T_INTEGER,
             "H5AD sparse index dataset must have an integer dtype");

    std::vector<std::uint64_t> out(count, 0);
    if (count == 0) {
        return out;
    }
    if (H5Tget_sign(type.get()) == H5T_SGN_NONE) {
        std::vector<unsigned long long> values(count, 0);
        check_h5(H5Dread(dataset, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, values.data()) >= 0,
                 "Failed to read unsigned HDF5 sparse index dataset");
        std::copy(values.begin(), values.end(), out.begin());
    } else {
        std::vector<long long> values(count, 0);
        check_h5(H5Dread(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, values.data()) >= 0,
                 "Failed to read signed HDF5 sparse index dataset");
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < 0) {
                throw std::runtime_error("H5AD sparse index dataset contains a negative value");
            }
            out[i] = static_cast<std::uint64_t>(values[i]);
        }
    }
    return out;
}

struct DatasetLayout {
    bool chunked = false;
    bool filtered = false;
    std::vector<hsize_t> chunks;
};

DatasetLayout inspect_layout(hid_t dataset) {
    DatasetLayout out;
    H5Property dcpl(H5Dget_create_plist(dataset));
    check_h5(static_cast<bool>(dcpl), "Failed to inspect HDF5 dataset creation properties");
    const H5D_layout_t layout = H5Pget_layout(dcpl.get());
    out.chunked = layout == H5D_CHUNKED;
    const int filters = H5Pget_nfilters(dcpl.get());
    check_h5(filters >= 0, "Failed to inspect HDF5 filter pipeline");
    out.filtered = filters > 0;
    if (out.chunked) {
        const auto rank = static_cast<int>(dataset_shape(dataset).size());
        out.chunks.resize(static_cast<std::size_t>(rank), 0);
        check_h5(H5Pget_chunk(dcpl.get(), rank, out.chunks.data()) == rank,
                 "Failed to inspect HDF5 chunk dimensions");
    }
    return out;
}

std::string dtype_name(hid_t dataset) {
    H5Type type(H5Dget_type(dataset));
    check_h5(static_cast<bool>(type), "Failed to inspect HDF5 dataset type");
    const H5T_class_t cls = H5Tget_class(type.get());
    const std::size_t bits = H5Tget_precision(type.get());
    if (cls == H5T_INTEGER) {
        return std::string(H5Tget_sign(type.get()) == H5T_SGN_NONE ? "uint" : "int") +
               std::to_string(bits);
    }
    if (cls == H5T_FLOAT) {
        return "float" + std::to_string(bits);
    }
    switch (cls) {
        case H5T_STRING:
            return "string";
        case H5T_COMPOUND:
            return "compound";
        case H5T_ENUM:
            return "enum";
        case H5T_ARRAY:
            return "array";
        case H5T_VLEN:
            return "vlen";
        case H5T_OPAQUE:
            return "opaque";
        case H5T_REFERENCE:
            return "reference";
        default:
            return "unknown";
    }
}

std::string layout_name(H5D_layout_t layout) {
    switch (layout) {
        case H5D_COMPACT:
            return "compact";
        case H5D_CONTIGUOUS:
            return "contiguous";
        case H5D_CHUNKED:
            return "chunked";
#if H5_VERSION_GE(1, 10, 0)
        case H5D_VIRTUAL:
            return "virtual";
#endif
        default:
            return "unknown";
    }
}

struct InspectedFilter {
    H5Z_filter_t id = H5Z_FILTER_ERROR;
    unsigned int flags = 0;
    std::string name;
    std::vector<unsigned int> client_data;
    unsigned int config = 0;
};

InspectedFilter inspect_filter(hid_t dcpl, unsigned int index) {
    InspectedFilter out;
    std::vector<unsigned int> parameters(32, 0);
    std::size_t parameter_count = parameters.size();
    char name[128] = {0};
    out.id = H5Pget_filter2(
        dcpl, index, &out.flags, &parameter_count, parameters.data(),
        sizeof(name), name, &out.config);
    check_h5(out.id >= 0, "Failed to inspect HDF5 dataset filter");
    if (parameter_count > parameters.size()) {
        parameters.assign(parameter_count, 0);
        std::size_t retry_count = parameters.size();
        out.id = H5Pget_filter2(
            dcpl, index, &out.flags, &retry_count, parameters.data(),
            sizeof(name), name, &out.config);
        check_h5(out.id >= 0 && retry_count <= parameters.size(),
                 "Failed to read complete HDF5 filter parameters");
        parameter_count = retry_count;
    }
    parameters.resize(parameter_count);
    out.name = name;
    out.client_data = std::move(parameters);
    return out;
}

actionet::h5ad::DatasetInfo inspect_dataset_info(
    hid_t dataset,
    const std::string& name) {
    actionet::h5ad::DatasetInfo out;
    out.name = name;
    out.dtype = dtype_name(dataset);

    const auto dimensions = dataset_shape(dataset);
    out.shape.assign(dimensions.begin(), dimensions.end());
    H5Type type(H5Dget_type(dataset));
    check_h5(static_cast<bool>(type), "Failed to inspect HDF5 dataset type");
    std::uint64_t elements = 1;
    for (const auto dimension : dimensions) {
        if (dimension != 0 &&
            elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw std::runtime_error("HDF5 dataset logical size exceeds uint64 range");
        }
        elements *= static_cast<std::uint64_t>(dimension);
    }
    const std::size_t item_size = H5Tget_size(type.get());
    if (item_size != 0 &&
        elements > std::numeric_limits<std::uint64_t>::max() / item_size) {
        throw std::runtime_error("HDF5 dataset logical byte size exceeds uint64 range");
    }
    out.logical_bytes = elements * item_size;
    out.stored_bytes = H5Dget_storage_size(dataset);

    H5Property dcpl(H5Dget_create_plist(dataset));
    check_h5(static_cast<bool>(dcpl), "Failed to inspect HDF5 dataset properties");
    const H5D_layout_t layout = H5Pget_layout(dcpl.get());
    out.layout = layout_name(layout);
    if (layout == H5D_CHUNKED) {
        std::vector<hsize_t> chunks(dimensions.size(), 0);
        check_h5(H5Pget_chunk(
                     dcpl.get(), static_cast<int>(chunks.size()), chunks.data()) ==
                     static_cast<int>(chunks.size()),
                 "Failed to inspect HDF5 chunk dimensions");
        out.chunks.assign(chunks.begin(), chunks.end());
    }

    const int filter_count = H5Pget_nfilters(dcpl.get());
    check_h5(filter_count >= 0, "Failed to count HDF5 dataset filters");
    out.filters.reserve(static_cast<std::size_t>(filter_count));
    for (int index = 0; index < filter_count; ++index) {
        const auto filter =
            inspect_filter(dcpl.get(), static_cast<unsigned int>(index));

        actionet::h5ad::FilterInfo filter_info;
        filter_info.id = static_cast<std::uint32_t>(filter.id);
        filter_info.flags = filter.flags;
        filter_info.name = filter.name;
        filter_info.client_data.assign(
            filter.client_data.begin(), filter.client_data.end());
        unsigned int available_config = 0;
        if (H5Zfilter_avail(filter.id) > 0 &&
            H5Zget_filter_info(filter.id, &available_config) >= 0) {
            filter_info.decode_available =
                (available_config & H5Z_FILTER_CONFIG_DECODE_ENABLED) != 0;
            filter_info.encode_available =
                (available_config & H5Z_FILTER_CONFIG_ENCODE_ENABLED) != 0;
        }
        out.filters.push_back(std::move(filter_info));
    }
    return out;
}

void append_dataset_info(actionet::h5ad::MatrixInfo& matrix,
                         actionet::h5ad::DatasetInfo dataset) {
    if (matrix.logical_bytes >
            std::numeric_limits<std::uint64_t>::max() - dataset.logical_bytes ||
        matrix.stored_bytes >
            std::numeric_limits<std::uint64_t>::max() - dataset.stored_bytes) {
        throw std::runtime_error("H5AD matrix byte inventory exceeds uint64 range");
    }
    matrix.logical_bytes += dataset.logical_bytes;
    matrix.stored_bytes += dataset.stored_bytes;
    matrix.datasets.push_back(std::move(dataset));
}

void reset_dataset_inventory(actionet::h5ad::MatrixInfo& matrix) {
    matrix.logical_bytes = 0;
    matrix.stored_bytes = 0;
    matrix.datasets.clear();
}

void copy_filters(hid_t source_dataset, hid_t destination_dcpl) {
    H5Property source_dcpl(H5Dget_create_plist(source_dataset));
    check_h5(static_cast<bool>(source_dcpl), "Failed to inspect source HDF5 filters");
    const int nfilters = H5Pget_nfilters(source_dcpl.get());
    check_h5(nfilters >= 0, "Failed to count source HDF5 filters");
    for (int index = 0; index < nfilters; ++index) {
        const auto filter =
            inspect_filter(source_dcpl.get(), static_cast<unsigned int>(index));
        unsigned int available_config = 0;
        if (H5Zfilter_avail(filter.id) <= 0 ||
            H5Zget_filter_info(filter.id, &available_config) < 0 ||
            (available_config & H5Z_FILTER_CONFIG_ENCODE_ENABLED) == 0) {
            throw std::runtime_error(
                "Cannot preserve HDF5 filter '" + filter.name +
                "' because its encoder is unavailable");
        }
        check_h5(H5Pset_filter(
                     destination_dcpl, filter.id, filter.flags,
                     filter.client_data.size(), filter.client_data.data()) >= 0,
                 "Failed to apply destination HDF5 filter");
    }
}

void validate_filter_capability(hid_t source_dataset, bool require_encoder) {
    H5Property source_dcpl(H5Dget_create_plist(source_dataset));
    check_h5(static_cast<bool>(source_dcpl), "Failed to inspect source HDF5 filters");
    const int nfilters = H5Pget_nfilters(source_dcpl.get());
    check_h5(nfilters >= 0, "Failed to count source HDF5 filters");
    for (int index = 0; index < nfilters; ++index) {
        const auto filter =
            inspect_filter(source_dcpl.get(), static_cast<unsigned int>(index));
        unsigned int available_config = 0;
        const bool available =
            H5Zfilter_avail(filter.id) > 0 &&
            H5Zget_filter_info(filter.id, &available_config) >= 0;
        const bool can_decode =
            available &&
            (available_config & H5Z_FILTER_CONFIG_DECODE_ENABLED) != 0;
        const bool can_encode =
            available &&
            (available_config & H5Z_FILTER_CONFIG_ENCODE_ENABLED) != 0;
        if (!can_decode || (require_encoder && !can_encode)) {
            throw std::runtime_error(
                "HDF5 filter '" + filter.name +
                "' is unavailable for native " +
                (require_encoder ? "decode/encode" : "decode"));
        }
    }
}

H5Property dataset_dcpl(hid_t source_dataset,
                        const std::vector<hsize_t>& output_dims,
                        bool extendable,
                        actionet::h5ad::LayoutPolicy policy) {
    H5Property dcpl(H5Pcreate(H5P_DATASET_CREATE));
    check_h5(static_cast<bool>(dcpl), "Failed to create destination HDF5 DCPL");
    const DatasetLayout source_layout = inspect_layout(source_dataset);
    const bool needs_chunks = extendable || source_layout.chunked ||
                              (policy == actionet::h5ad::LayoutPolicy::Preserve &&
                               source_layout.filtered);
    if (needs_chunks) {
        std::vector<hsize_t> chunks(output_dims.size(), 1);
        if (source_layout.chunked && source_layout.chunks.size() == output_dims.size()) {
            chunks = source_layout.chunks;
            // Cap an oversized inherited 1-D chunk. Some sources store huge
            // chunks (e.g. 10M elements); reusing them for a smaller subset
            // over-allocates a trailing chunk and concentrates dirty pages.
            // Only shrink toward a byte-budget target; never grow, so filtered
            // sources keep chunk boundaries their codecs expect.
            if (output_dims.size() == 1) {
                std::size_t element_size = 0;
                {
                    H5Type source_type(H5Dget_type(source_dataset));
                    if (source_type) {
                        element_size = H5Tget_size(source_type.get());
                    }
                }
                if (element_size == 0) {
                    element_size = 1;
                }
                constexpr hsize_t kMaxChunkBytes = 32ULL * 1024ULL * 1024ULL;
                hsize_t max_elements = kMaxChunkBytes /
                    static_cast<hsize_t>(element_size);
                if (max_elements == 0) {
                    max_elements = 1;
                }
                // Do not cap filtered chunks: their compressed layout is tied
                // to the exact chunk shape and the encoder was validated for it.
                if (!source_layout.filtered && chunks[0] > max_elements) {
                    chunks[0] = max_elements;
                }
            }
        } else if (!output_dims.empty()) {
            if (output_dims.size() == 1) {
                // A contiguous source gives no chunk hint. A tiny fixed chunk
                // (the former 65536-element default) builds an enormous chunk
                // B-tree over billions of NNZ, while blindly inheriting a huge
                // source chunk over-allocates a trailing chunk. Size the chunk
                // to a fixed target byte budget derived from the element width
                // so the B-tree stays small and writeback is contiguous.
                std::size_t element_size = 0;
                {
                    H5Type source_type(H5Dget_type(source_dataset));
                    if (source_type) {
                        element_size = H5Tget_size(source_type.get());
                    }
                }
                if (element_size == 0) {
                    element_size = 1;
                }
                constexpr hsize_t kTargetChunkBytes = 16ULL * 1024ULL * 1024ULL;
                hsize_t elements = kTargetChunkBytes /
                    static_cast<hsize_t>(element_size);
                if (elements == 0) {
                    elements = 1;
                }
                const hsize_t bound = output_dims[0] == 0 ? elements
                                                          : output_dims[0];
                chunks[0] = std::max<hsize_t>(1, std::min<hsize_t>(elements, bound));
            } else {
                chunks[0] = std::max<hsize_t>(1, std::min<hsize_t>(
                    output_dims[0] == 0 ? 1024 : output_dims[0], 1024));
                for (std::size_t axis = 1; axis < output_dims.size(); ++axis) {
                    chunks[axis] = std::max<hsize_t>(1, output_dims[axis]);
                }
            }
        }
        for (std::size_t axis = 0; axis < chunks.size(); ++axis) {
            if (output_dims[axis] > 0) {
                chunks[axis] = std::max<hsize_t>(1, std::min(chunks[axis], output_dims[axis]));
            } else {
                chunks[axis] = std::max<hsize_t>(1, chunks[axis]);
            }
        }
        check_h5(H5Pset_chunk(dcpl.get(), static_cast<int>(chunks.size()), chunks.data()) >= 0,
                 "Failed to configure destination HDF5 chunks");
    }
    if (policy == actionet::h5ad::LayoutPolicy::Preserve && source_layout.filtered) {
        copy_filters(source_dataset, dcpl.get());
    }
    return dcpl;
}

H5Group ensure_group(hid_t file, const std::string& raw_path) {
    const std::string path = raw_path.empty() ? "/" :
        (raw_path.front() == '/' ? raw_path : "/" + raw_path);
    if (path == "/") {
        return H5Group(H5Gopen2(file, "/", H5P_DEFAULT));
    }
    H5Group current(H5Gopen2(file, "/", H5P_DEFAULT));
    check_h5(static_cast<bool>(current), "Failed to open HDF5 root group");
    std::size_t start = 1;
    while (start < path.size()) {
        const auto slash = path.find('/', start);
        const std::string part = path.substr(start, slash - start);
        if (!part.empty()) {
            hid_t next = -1;
            if (H5Lexists(current.get(), part.c_str(), H5P_DEFAULT) > 0) {
                next = H5Gopen2(current.get(), part.c_str(), H5P_DEFAULT);
            } else {
                next = H5Gcreate2(current.get(), part.c_str(), H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            }
            check_h5(next >= 0, "Failed to create destination HDF5 parent group");
            current.reset(next);
        }
        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }
    return current;
}

void delete_if_exists(hid_t file, const std::string& raw_path) {
    const std::string path = normalize_h5_path(raw_path);
    if (H5Lexists(file, path.c_str(), H5P_DEFAULT) > 0) {
        check_h5(H5Ldelete(file, path.c_str(), H5P_DEFAULT) >= 0,
                 "Failed to replace destination HDF5 object");
    }
}

H5Dataset open_dataset(hid_t parent, const char* name) {
    H5Dataset dataset(H5Dopen2(parent, name, H5P_DEFAULT));
    if (!dataset) {
        throw std::runtime_error(std::string("Missing H5AD matrix dataset '") + name + "'");
    }
    return dataset;
}

struct OpenMatrix {
    actionet::h5ad::MatrixInfo info;
    H5Group group;
    H5Dataset object;
    H5Dataset data;
    H5Dataset indices;
    H5Dataset indptr;
};

OpenMatrix open_matrix(hid_t file, const std::string& raw_path, bool structural_check) {
    using actionet::h5ad::MatrixEncoding;

    const std::string path = normalize_h5_path(raw_path);
    H5O_info_t object_info;
#if H5_VERSION_GE(1, 12, 0)
    check_h5(H5Oget_info_by_name(file, path.c_str(), &object_info,
                                 H5O_INFO_BASIC, H5P_DEFAULT) >= 0,
             "H5AD matrix path does not exist");
#else
    check_h5(H5Oget_info_by_name(file, path.c_str(), &object_info,
                                 H5P_DEFAULT) >= 0,
             "H5AD matrix path does not exist");
#endif

    OpenMatrix out;
    if (object_info.type == H5O_TYPE_DATASET) {
        out.object.reset(H5Dopen2(file, path.c_str(), H5P_DEFAULT));
        check_h5(static_cast<bool>(out.object), "Failed to open dense H5AD matrix");
        const auto dims = dataset_shape(out.object.get());
        if (dims.size() != 2) {
            throw std::runtime_error("Dense H5AD matrix must be two-dimensional");
        }
        require_numeric_type(out.object.get(), "array");
        out.info.encoding = MatrixEncoding::Dense;
        out.info.rows = dims[0];
        out.info.cols = dims[1];
        out.info.nnz = checked_product(dims[0], dims[1], "Dense H5AD element count");
        H5Type type(H5Dget_type(out.object.get()));
        out.info.data_item_size = H5Tget_size(type.get());
        out.info.encoding_type = read_string_attr(out.object.get(), "encoding-type");
        out.info.encoding_version = read_string_attr(out.object.get(), "encoding-version");
        if (out.info.encoding_type != "array" || out.info.encoding_version != "0.2.0") {
            throw std::runtime_error(
                "Unsupported dense H5AD encoding; expected array 0.2.0");
        }
        const auto layout = inspect_layout(out.object.get());
        out.info.chunked = layout.chunked;
        out.info.filtered = layout.filtered;
        append_dataset_info(
            out.info, inspect_dataset_info(out.object.get(), "data"));
        return out;
    }

    if (object_info.type != H5O_TYPE_GROUP) {
        throw std::runtime_error("H5AD matrix must be a dataset or sparse group");
    }
    out.group.reset(H5Gopen2(file, path.c_str(), H5P_DEFAULT));
    check_h5(static_cast<bool>(out.group), "Failed to open sparse H5AD matrix");
    out.info.encoding_type = read_string_attr(out.group.get(), "encoding-type");
    out.info.encoding_version = read_string_attr(out.group.get(), "encoding-version");
    if (out.info.encoding_version != "0.1.0") {
        throw std::runtime_error(
            "Unsupported sparse H5AD encoding version; expected 0.1.0");
    }
    if (out.info.encoding_type == "csr_matrix") {
        out.info.encoding = MatrixEncoding::CSR;
    } else if (out.info.encoding_type == "csc_matrix") {
        out.info.encoding = MatrixEncoding::CSC;
    } else {
        throw std::runtime_error(
            "Unsupported sparse H5AD encoding; expected csr_matrix or csc_matrix");
    }

    const auto shape = read_shape_attr(out.group.get());
    out.info.rows = shape[0];
    out.info.cols = shape[1];
    out.data = open_dataset(out.group.get(), "data");
    out.indices = open_dataset(out.group.get(), "indices");
    out.indptr = open_dataset(out.group.get(), "indptr");
    require_numeric_type(out.data.get(), "data");
    require_integer_type(out.indices.get(), "indices");
    require_integer_type(out.indptr.get(), "indptr");
    const auto data_dims = dataset_shape(out.data.get());
    const auto indices_dims = dataset_shape(out.indices.get());
    const auto indptr_dims = dataset_shape(out.indptr.get());
    if (data_dims.size() != 1 || indices_dims.size() != 1 || indptr_dims.size() != 1) {
        throw std::runtime_error("H5AD sparse payload datasets must be one-dimensional");
    }
    if (data_dims[0] != indices_dims[0]) {
        throw std::runtime_error("H5AD sparse data and indices lengths differ");
    }
    const std::uint64_t major = out.info.encoding == MatrixEncoding::CSR
        ? out.info.rows : out.info.cols;
    if (indptr_dims[0] != major + 1) {
        throw std::runtime_error("H5AD sparse indptr length does not match matrix shape");
    }
    out.info.nnz = data_dims[0];
    H5Type data_type(H5Dget_type(out.data.get()));
    H5Type indices_type(H5Dget_type(out.indices.get()));
    H5Type indptr_type(H5Dget_type(out.indptr.get()));
    out.info.data_item_size = H5Tget_size(data_type.get());
    out.info.indices_item_size = H5Tget_size(indices_type.get());
    out.info.indptr_item_size = H5Tget_size(indptr_type.get());
    const auto data_layout = inspect_layout(out.data.get());
    const auto indices_layout = inspect_layout(out.indices.get());
    const auto indptr_layout = inspect_layout(out.indptr.get());
    out.info.chunked = data_layout.chunked || indices_layout.chunked ||
                       indptr_layout.chunked;
    out.info.filtered = data_layout.filtered || indices_layout.filtered ||
                        indptr_layout.filtered;
    append_dataset_info(
        out.info, inspect_dataset_info(out.data.get(), "data"));
    append_dataset_info(
        out.info, inspect_dataset_info(out.indices.get(), "indices"));
    append_dataset_info(
        out.info, inspect_dataset_info(out.indptr.get(), "indptr"));

    if (structural_check) {
        const auto indptr = read_integer_1d(out.indptr.get());
        if (indptr.empty() || indptr.front() != 0 || indptr.back() != out.info.nnz) {
            throw std::runtime_error("H5AD sparse indptr endpoints are invalid");
        }
        for (std::size_t i = 1; i < indptr.size(); ++i) {
            if (indptr[i] < indptr[i - 1]) {
                throw std::runtime_error("H5AD sparse indptr is not monotonic");
            }
        }
    }
    return out;
}

void validate_transfer_filters(
    const OpenMatrix& source,
    actionet::h5ad::LayoutPolicy policy) {
    const bool require_encoder =
        policy == actionet::h5ad::LayoutPolicy::Preserve;
    if (source.info.encoding == actionet::h5ad::MatrixEncoding::Dense) {
        validate_filter_capability(source.object.get(), require_encoder);
        return;
    }
    validate_filter_capability(source.data.get(), require_encoder);
    validate_filter_capability(source.indices.get(), require_encoder);
    validate_filter_capability(source.indptr.get(), require_encoder);
}

std::vector<std::uint64_t> resolve_axis(
    const actionet::h5ad::AxisSelection& selection,
    std::uint64_t size) {
    if (!selection.indices.has_value()) {
        std::vector<std::uint64_t> out(static_cast<std::size_t>(size));
        std::iota(out.begin(), out.end(), std::uint64_t{0});
        return out;
    }
    std::vector<std::uint64_t> out = *selection.indices;
    for (const auto value : out) {
        if (value >= size) {
            throw std::out_of_range("H5AD matrix selection index is out of bounds");
        }
    }
    return out;
}

bool strictly_increasing(const std::vector<std::uint64_t>& values) {
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] <= values[i - 1]) {
            return false;
        }
    }
    return true;
}

bool is_identity(const actionet::h5ad::AxisSelection& selection,
                 std::uint64_t size) {
    if (!selection.indices.has_value()) {
        return true;
    }
    const auto& values = *selection.indices;
    if (values.size() != static_cast<std::size_t>(size)) {
        return false;
    }
    for (std::uint64_t i = 0; i < size; ++i) {
        if (values[static_cast<std::size_t>(i)] != i) {
            return false;
        }
    }
    return true;
}

std::uint64_t checked_bytes(std::uint64_t count, std::size_t item_size) {
    if (item_size != 0 && count > std::numeric_limits<std::uint64_t>::max() / item_size) {
        throw std::runtime_error("H5AD transfer byte count overflow");
    }
    return count * static_cast<std::uint64_t>(item_size);
}

struct ReadSpan {
    std::uint64_t major_start = 0;
    std::uint64_t major_end = 0;
    std::uint64_t element_start = 0;
    std::uint64_t element_end = 0;
    std::size_t selected_begin = 0;
    std::size_t selected_end = 0;
};

std::uint64_t span_bytes(const ReadSpan& span, std::size_t bytes_per_element) {
    return checked_bytes(span.element_end - span.element_start, bytes_per_element);
}

std::vector<ReadSpan> build_compressed_spans(
    const std::vector<std::uint64_t>& indptr,
    const std::vector<std::uint64_t>& selected,
    std::size_t buffer_bytes_per_element,
    std::size_t gap_bytes_per_element,
    const std::vector<std::size_t>& chunk_element_sizes,
    const actionet::h5ad::TransferOptions& options,
    bool force_scan) {
    std::vector<ReadSpan> base;
    if (selected.empty()) {
        return base;
    }
    const std::uint64_t max_bytes = std::max<std::uint64_t>(1, options.max_buffer_bytes);
    const std::size_t max_major = std::max<std::size_t>(1, options.max_rows_per_batch);

    std::size_t begin = 0;
    while (begin < selected.size()) {
        std::size_t end = begin + 1;
        const std::uint64_t major_start = selected[begin];
        std::uint64_t major_end = major_start + 1;
        while (end < selected.size() &&
               selected[end] == selected[end - 1] + 1 &&
               end - begin < max_major) {
            const std::uint64_t candidate_end = selected[end] + 1;
            const std::uint64_t candidate_bytes =
                checked_bytes(indptr[candidate_end] - indptr[major_start],
                              buffer_bytes_per_element);
            if (candidate_bytes > max_bytes && end > begin) {
                break;
            }
            major_end = candidate_end;
            ++end;
        }
        base.push_back(ReadSpan{
            major_start,
            major_end,
            indptr[major_start],
            indptr[major_end],
            begin,
            end,
        });
        begin = end;
    }

    std::vector<ReadSpan> merged;
    merged.reserve(base.size());
    for (const auto& next : base) {
        if (merged.empty()) {
            merged.push_back(next);
            continue;
        }
        auto& current = merged.back();
        const std::uint64_t combined_bytes =
            checked_bytes(
                next.element_end - current.element_start,
                buffer_bytes_per_element);
        const std::uint64_t gap_bytes =
            checked_bytes(
                next.element_start - current.element_end,
                gap_bytes_per_element);
        const bool same_chunk =
            current.element_end > current.element_start &&
            next.element_start > 0 &&
            std::any_of(
                chunk_element_sizes.begin(), chunk_element_sizes.end(),
                [&](std::size_t chunk_elements) {
                    return chunk_elements > 0 &&
                        (current.element_end - 1) / chunk_elements ==
                            next.element_start / chunk_elements;
                });
        const bool selected_limit =
            next.selected_end - current.selected_begin <= max_major;
        const bool merge = selected_limit && combined_bytes <= max_bytes &&
            (force_scan || gap_bytes <= options.gap_merge_bytes || same_chunk);
        if (merge) {
            current.major_end = next.major_end;
            current.element_end = next.element_end;
            current.selected_end = next.selected_end;
        } else {
            merged.push_back(next);
        }
    }
    return merged;
}

std::uint64_t spans_cost(const std::vector<ReadSpan>& spans,
                         std::size_t bytes_per_element,
                         std::size_t call_penalty,
                         std::uint64_t calls_per_span) {
    std::uint64_t cost = 0;
    for (const auto& span : spans) {
        cost += span_bytes(span, bytes_per_element);
    }
    const std::uint64_t call_count =
        static_cast<std::uint64_t>(spans.size()) * calls_per_span;
    const std::uint64_t penalty = checked_bytes(call_count, call_penalty);
    return cost > std::numeric_limits<std::uint64_t>::max() - penalty
        ? std::numeric_limits<std::uint64_t>::max()
        : cost + penalty;
}

std::vector<std::size_t> chunk_element_sizes(hid_t first, hid_t second) {
    std::vector<std::size_t> out;
    for (const hid_t dataset : {first, second}) {
        const auto layout = inspect_layout(dataset);
        if (layout.chunked && layout.chunks.size() == 1) {
            const auto size = static_cast<std::size_t>(layout.chunks[0]);
            if (std::find(out.begin(), out.end(), size) == out.end()) {
                out.push_back(size);
            }
        }
    }
    return out;
}

void read_1d_raw(hid_t dataset, hid_t mem_type,
                 std::uint64_t start, std::uint64_t count,
                 void* destination) {
    if (count == 0) {
        return;
    }
    const hsize_t starts[1] = {static_cast<hsize_t>(start)};
    const hsize_t counts[1] = {static_cast<hsize_t>(count)};
    H5Space file_space(H5Dget_space(dataset));
    check_h5(static_cast<bool>(file_space), "Failed to open HDF5 source dataspace");
    check_h5(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET,
                                 starts, nullptr, counts, nullptr) >= 0,
             "Failed to select HDF5 source hyperslab");
    H5Space mem_space(H5Screate_simple(1, counts, nullptr));
    check_h5(static_cast<bool>(mem_space), "Failed to create HDF5 memory dataspace");
    check_h5(H5Dread(dataset, mem_type, mem_space.get(), file_space.get(),
                     H5P_DEFAULT, destination) >= 0,
             "Failed to read HDF5 dataset span");
}

std::vector<std::uint64_t> read_1d_indices(
    hid_t dataset, std::uint64_t start, std::uint64_t count) {
    H5Type type(H5Dget_type(dataset));
    check_h5(static_cast<bool>(type), "Failed to inspect sparse indices dtype");
    std::vector<std::uint64_t> out(static_cast<std::size_t>(count), 0);
    if (count == 0) {
        return out;
    }
    if (H5Tget_sign(type.get()) == H5T_SGN_NONE) {
        std::vector<unsigned long long> values(static_cast<std::size_t>(count), 0);
        read_1d_raw(dataset, H5T_NATIVE_ULLONG, start, count, values.data());
        std::copy(values.begin(), values.end(), out.begin());
    } else {
        std::vector<long long> values(static_cast<std::size_t>(count), 0);
        read_1d_raw(dataset, H5T_NATIVE_LLONG, start, count, values.data());
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i] < 0) {
                throw std::runtime_error("Sparse indices contain a negative value");
            }
            out[i] = static_cast<std::uint64_t>(values[i]);
        }
    }
    return out;
}

H5Dataset create_1d_dataset(
    hid_t parent,
    const char* name,
    hid_t file_type,
    hid_t source_dataset,
    std::uint64_t initial_size,
    bool extendable,
    actionet::h5ad::LayoutPolicy policy) {
    const hsize_t dims[1] = {static_cast<hsize_t>(initial_size)};
    const hsize_t maxdims[1] = {extendable ? H5S_UNLIMITED :
                                static_cast<hsize_t>(initial_size)};
    H5Space space(H5Screate_simple(1, dims, extendable ? maxdims : nullptr));
    check_h5(static_cast<bool>(space), "Failed to create destination HDF5 dataspace");
    H5Property dcpl = dataset_dcpl(source_dataset, {dims[0]}, extendable, policy);
    H5Dataset dataset(H5Dcreate2(parent, name, file_type, space.get(), H5P_DEFAULT,
                                 dcpl.get(), H5P_DEFAULT));
    check_h5(static_cast<bool>(dataset), "Failed to create destination HDF5 dataset");
    return dataset;
}

void write_1d(hid_t dataset, hid_t mem_type,
              std::uint64_t offset, std::uint64_t count,
              const void* source, bool extendable) {
    if (count == 0) {
        return;
    }
    if (extendable) {
        const hsize_t next_dims[1] = {static_cast<hsize_t>(offset + count)};
        check_h5(H5Dset_extent(dataset, next_dims) >= 0,
                 "Failed to extend destination HDF5 dataset");
    }
    const hsize_t starts[1] = {static_cast<hsize_t>(offset)};
    const hsize_t counts[1] = {static_cast<hsize_t>(count)};
    H5Space file_space(H5Dget_space(dataset));
    check_h5(static_cast<bool>(file_space), "Failed to open destination HDF5 dataspace");
    check_h5(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET,
                                 starts, nullptr, counts, nullptr) >= 0,
             "Failed to select destination HDF5 hyperslab");
    H5Space mem_space(H5Screate_simple(1, counts, nullptr));
    check_h5(static_cast<bool>(mem_space), "Failed to create destination memory dataspace");
    check_h5(H5Dwrite(dataset, mem_type, mem_space.get(), file_space.get(),
                      H5P_DEFAULT, source) >= 0,
             "Failed to write destination HDF5 dataset");
}

void append_raw(std::vector<unsigned char>& destination,
                const unsigned char* source,
                std::size_t bytes) {
    const std::size_t old = destination.size();
    destination.resize(old + bytes);
    if (bytes > 0) {
        std::memcpy(destination.data() + old, source, bytes);
    }
}

actionet::h5ad::TransferStats transfer_compressed(
    const OpenMatrix& source,
    hid_t destination_file,
    const std::string& destination_path,
    const std::vector<std::uint64_t>& major_selection,
    const actionet::h5ad::AxisSelection& minor_axis_selection,
    const actionet::h5ad::TransferOptions& options) {
    using actionet::h5ad::MatrixEncoding;
    using actionet::h5ad::TransferStats;

    const auto planning_started = Clock::now();
    const bool ordered_unique =
        major_selection.size() < 2 || strictly_increasing(major_selection);
    const bool is_csr = source.info.encoding == MatrixEncoding::CSR;
    const std::uint64_t source_minor = is_csr ? source.info.cols : source.info.rows;
    const auto minor_selection = resolve_axis(minor_axis_selection, source_minor);
    const bool minor_identity = is_identity(minor_axis_selection, source_minor);
    const bool minor_empty = minor_selection.empty();
    const std::uint64_t output_rows = is_csr
        ? static_cast<std::uint64_t>(major_selection.size())
        : static_cast<std::uint64_t>(minor_selection.size());
    const std::uint64_t output_cols = is_csr
        ? static_cast<std::uint64_t>(minor_selection.size())
        : static_cast<std::uint64_t>(major_selection.size());

    const auto indptr = read_integer_1d(source.indptr.get());
    const std::size_t bytes_per_source_element =
        source.info.data_item_size + source.info.indices_item_size;
    const std::size_t bytes_per_source_buffer_element =
        source.info.data_item_size +
        std::max<std::size_t>(source.info.indices_item_size, sizeof(std::uint64_t));
    const auto physical_chunk_elements =
        chunk_element_sizes(source.data.get(), source.indices.get());
    actionet::h5ad::TransferOptions bounded_options = options;
    bounded_options.max_buffer_bytes =
        std::max<std::size_t>(
            1, options.max_buffer_bytes / (ordered_unique ? 2 : 3));

    std::vector<ReadSpan> spans;
    if (ordered_unique) {
        auto gather_spans = build_compressed_spans(
            indptr, major_selection,
            bytes_per_source_buffer_element, bytes_per_source_element,
            physical_chunk_elements, bounded_options, false);
        auto scan_spans = build_compressed_spans(
            indptr, major_selection,
            bytes_per_source_buffer_element, bytes_per_source_element,
            physical_chunk_elements, bounded_options, true);
        const auto gather_cost = spans_cost(
            gather_spans, bytes_per_source_element,
            options.gap_merge_bytes, 2);
        const auto scan_cost = spans_cost(
            scan_spans, bytes_per_source_element,
            options.gap_merge_bytes, 2);
        spans = scan_cost < gather_cost
            ? std::move(scan_spans) : std::move(gather_spans);
    } else {
        // General selectors are planned per bounded output batch below.
    }

    std::uint64_t selected_elements = 0;
    for (const auto major : major_selection) {
        selected_elements += indptr[major + 1] - indptr[major];
    }

    TransferStats stats;
    stats.source = source.info;
    stats.selected_source_bytes = minor_empty
        ? 0 : checked_bytes(selected_elements, bytes_per_source_element);
    stats.planning_seconds = elapsed_seconds(planning_started);
    stats.span_count = 0;

    std::vector<std::vector<std::uint64_t>> minor_map;
    if (!minor_identity && !minor_empty) {
        minor_map.resize(static_cast<std::size_t>(source_minor));
        for (std::uint64_t output_index = 0;
             output_index < minor_selection.size(); ++output_index) {
            minor_map[static_cast<std::size_t>(
                minor_selection[static_cast<std::size_t>(output_index)])]
                .push_back(output_index);
        }
    }

    delete_if_exists(destination_file, destination_path);
    const auto [parent_path, object_name] = split_parent_name(destination_path);
    H5Group parent = ensure_group(destination_file, parent_path);
    H5Group destination_group(H5Gcreate2(
        parent.get(), object_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    check_h5(static_cast<bool>(destination_group),
             "Failed to create destination H5AD sparse group");
    write_string_attr(destination_group.get(), "encoding-type",
                      is_csr ? "csr_matrix" : "csc_matrix");
    write_string_attr(destination_group.get(), "encoding-version", "0.1.0");
    write_shape_attr(destination_group.get(), output_rows, output_cols);

    H5Type source_data_type(H5Dget_type(source.data.get()));
    check_h5(static_cast<bool>(source_data_type),
             "Failed to inspect source sparse data dtype");
    const bool output_indices_int32 =
        (is_csr ? output_cols : output_rows) <=
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    const hid_t output_indices_type =
        output_indices_int32 ? H5T_STD_I32LE : H5T_STD_I64LE;

    const bool fixed_payload = minor_identity;
    const std::uint64_t fixed_nnz = fixed_payload ? selected_elements : 0;
    H5Dataset destination_data = create_1d_dataset(
        destination_group.get(), "data", source_data_type.get(), source.data.get(),
        fixed_nnz, !fixed_payload, options.layout_policy);
    H5Dataset destination_indices = create_1d_dataset(
        destination_group.get(), "indices", output_indices_type, source.indices.get(),
        fixed_nnz, !fixed_payload, options.layout_policy);

    H5Type source_indices_type(H5Dget_type(source.indices.get()));
    // Real AnnData files store indices/indptr as unsigned integers, so the
    // fast raw-append path must accept H5T_SGN_NONE sources as well as signed
    // ones. Correctness is preserved because minor_identity guarantees the
    // stored index values are unchanged, and the size guard below only enables
    // the fast path when the source item size matches the destination width
    // (item_size 4 pairs with an int32 destination, 8 with int64). When the
    // axis extent fits int32 (output_indices_int32) every valid unsigned index
    // is < 2^31, so the width-matched HDF5 conversion at write time is exact.
    const bool fast_indices =
        ordered_unique &&
        minor_identity &&
        H5Tget_class(source_indices_type.get()) == H5T_INTEGER &&
        ((output_indices_int32 && H5Tget_size(source_indices_type.get()) == 4) ||
         (!output_indices_int32 && H5Tget_size(source_indices_type.get()) == 8));

    std::vector<unsigned char> output_data;
    std::vector<unsigned char> output_indices_raw;
    std::vector<long long> output_indices_converted;
    const std::size_t output_buffer_budget = bounded_options.max_buffer_bytes;
    const std::size_t output_index_item_size =
        fast_indices ? source.info.indices_item_size : sizeof(long long);
    const std::size_t output_element_bytes =
        source.info.data_item_size + output_index_item_size;
    output_data.reserve(
        output_buffer_budget * source.info.data_item_size /
        std::max<std::size_t>(1, output_element_bytes));
    if (fast_indices) {
        output_indices_raw.reserve(
            output_buffer_budget * output_index_item_size /
            std::max<std::size_t>(1, output_element_bytes));
    } else {
        output_indices_converted.reserve(
            output_buffer_budget * output_index_item_size /
            std::max<std::size_t>(1, output_element_bytes) /
            sizeof(long long));
    }
    std::vector<std::uint64_t> output_indptr;
    output_indptr.reserve(major_selection.size() + 1);
    output_indptr.push_back(0);
    if (minor_empty) {
        output_indptr.resize(major_selection.size() + 1, 0);
        spans.clear();
    }
    std::uint64_t written_nnz = 0;
    std::uint64_t buffered_nnz = 0;
    std::uint64_t bytes_since_fsync = 0;

    auto flush_buffers = [&]() {
        if (buffered_nnz == 0) {
            return;
        }
        const auto started = Clock::now();
        write_1d(destination_data.get(), source_data_type.get(),
                 written_nnz, buffered_nnz, output_data.data(), !fixed_payload);
        if (fast_indices) {
            write_1d(destination_indices.get(), source_indices_type.get(),
                     written_nnz, buffered_nnz, output_indices_raw.data(), !fixed_payload);
        } else {
            write_1d(destination_indices.get(), H5T_NATIVE_LLONG,
                     written_nnz, buffered_nnz, output_indices_converted.data(),
                     !fixed_payload);
        }
        stats.destination_write_seconds += elapsed_seconds(started);
        stats.hdf5_write_calls += 2;
        bytes_since_fsync += checked_bytes(
            buffered_nnz,
            source.info.data_item_size + output_index_item_size);
        written_nnz += buffered_nnz;
        buffered_nnz = 0;
        output_data.clear();
        output_indices_raw.clear();
        output_indices_converted.clear();
        if (bytes_since_fsync >= kIncrementalFsyncBytes) {
            const auto fsync_started = Clock::now();
            flush_and_fsync_file(destination_file, true);
            stats.destination_fsync_seconds += elapsed_seconds(fsync_started);
            bytes_since_fsync = 0;
        }
    };

    const std::size_t data_item_size = source.info.data_item_size;
    const std::size_t source_index_item_size = source.info.indices_item_size;
    auto append_major = [&](
        std::uint64_t major,
        std::uint64_t source_element_start,
        const std::vector<unsigned char>& source_data,
        const std::vector<unsigned char>& source_indices_raw,
        const std::vector<std::uint64_t>& source_indices) {
        const std::uint64_t local_start =
            indptr[major] - source_element_start;
        const std::uint64_t local_end =
            indptr[major + 1] - source_element_start;
        const std::uint64_t row_count = local_end - local_start;

        if (minor_identity) {
            append_raw(
                output_data,
                source_data.data() + checked_bytes(local_start, data_item_size),
                static_cast<std::size_t>(
                    checked_bytes(row_count, data_item_size)));
            if (fast_indices) {
                append_raw(
                    output_indices_raw,
                    source_indices_raw.data() +
                        checked_bytes(local_start, source_index_item_size),
                    static_cast<std::size_t>(
                        checked_bytes(row_count, source_index_item_size)));
            } else {
                for (std::uint64_t p = local_start; p < local_end; ++p) {
                    const std::uint64_t index =
                        source_indices[static_cast<std::size_t>(p)];
                    if (index >= source_minor) {
                        throw std::runtime_error(
                            "H5AD sparse index exceeds minor-axis bounds");
                    }
                    output_indices_converted.push_back(
                        static_cast<long long>(index));
                }
            }
            buffered_nnz += row_count;
        } else {
            for (std::uint64_t p = local_start; p < local_end; ++p) {
                const std::uint64_t source_index =
                    source_indices[static_cast<std::size_t>(p)];
                if (source_index >= source_minor) {
                    throw std::runtime_error(
                        "H5AD sparse index exceeds minor-axis bounds");
                }
                for (const auto output_index :
                     minor_map[static_cast<std::size_t>(source_index)]) {
                    append_raw(
                        output_data,
                        source_data.data() + checked_bytes(p, data_item_size),
                        data_item_size);
                    output_indices_converted.push_back(
                        static_cast<long long>(output_index));
                    ++buffered_nnz;
                    const std::uint64_t current =
                        output_data.size() +
                        checked_bytes(
                            output_indices_converted.size(), sizeof(long long));
                    if (current >= output_buffer_budget) {
                        flush_buffers();
                    }
                }
            }
        }
        output_indptr.push_back(written_nnz + buffered_nnz);
        const std::uint64_t current_buffers =
            output_data.size() + output_indices_raw.size() +
            checked_bytes(output_indices_converted.size(), sizeof(long long));
        if (current_buffers >= output_buffer_budget) {
            flush_buffers();
        }
    };

    auto read_span = [&](const ReadSpan& span,
                         std::vector<unsigned char>& source_data,
                         std::vector<unsigned char>& source_indices_raw,
                         std::vector<std::uint64_t>& source_indices) {
        const std::uint64_t count = span.element_end - span.element_start;
        source_data.assign(
            static_cast<std::size_t>(checked_bytes(count, data_item_size)), 0);
        source_indices_raw.clear();
        source_indices.clear();

        const auto read_started = Clock::now();
        if (count > 0) {
            read_1d_raw(source.data.get(), source_data_type.get(),
                        span.element_start, count, source_data.data());
            if (fast_indices) {
                source_indices_raw.resize(
                    static_cast<std::size_t>(checked_bytes(
                        count, source_index_item_size)));
                read_1d_raw(source.indices.get(), source_indices_type.get(),
                            span.element_start, count, source_indices_raw.data());
            } else {
                source_indices = read_1d_indices(
                    source.indices.get(), span.element_start, count);
            }
            stats.hdf5_read_calls += 2;
        }
        const double source_read_seconds = elapsed_seconds(read_started);
        stats.source_read_seconds += source_read_seconds;
        const std::uint64_t bytes_read =
            checked_bytes(count, bytes_per_source_element);
        stats.source_bytes_read += bytes_read;
        ++stats.span_count;
        if (options.collect_span_stats) {
            stats.spans.push_back(actionet::h5ad::SpanStats{
                span.major_start,
                span.major_end,
                static_cast<std::uint64_t>(
                    span.selected_end - span.selected_begin),
                count,
                bytes_read,
                source_read_seconds,
                0.0,
            });
        }
    };

    for (const auto& span : spans) {
        std::vector<unsigned char> source_data;
        std::vector<unsigned char> source_indices_raw;
        std::vector<std::uint64_t> source_indices;
        read_span(
            span, source_data, source_indices_raw, source_indices);
        const auto pack_started = Clock::now();
        for (std::size_t selected_position = span.selected_begin;
             selected_position < span.selected_end; ++selected_position) {
            const std::uint64_t major = major_selection[selected_position];
            append_major(
                major, span.element_start, source_data,
                source_indices_raw, source_indices);
            const std::uint64_t current_buffers =
                output_data.size() + output_indices_raw.size() +
                checked_bytes(output_indices_converted.size(), sizeof(long long));
            const std::uint64_t source_buffers =
                source_data.size() + source_indices_raw.size() +
                checked_bytes(source_indices.size(), sizeof(std::uint64_t));
            stats.peak_buffer_bytes = std::max(
                stats.peak_buffer_bytes, source_buffers + current_buffers);
        }
        const double packing_seconds = elapsed_seconds(pack_started);
        stats.packing_seconds += packing_seconds;
        if (options.collect_span_stats && !stats.spans.empty()) {
            stats.spans.back().packing_seconds += packing_seconds;
        }
    }

    struct CachedMajor {
        std::vector<unsigned char> data;
        std::vector<std::uint64_t> indices;
    };
    for (std::size_t batch_begin = 0;
         !minor_empty && !ordered_unique &&
             batch_begin < major_selection.size();) {
        std::size_t batch_end = std::min(
            major_selection.size(),
            batch_begin + options.max_rows_per_batch);
        std::vector<std::uint64_t> unique;
        while (true) {
            unique.assign(
                major_selection.begin() + static_cast<std::ptrdiff_t>(batch_begin),
                major_selection.begin() + static_cast<std::ptrdiff_t>(batch_end));
            std::sort(unique.begin(), unique.end());
            unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
            std::uint64_t selected_buffer_bytes = 0;
            for (const auto major : unique) {
                selected_buffer_bytes += checked_bytes(
                    indptr[major + 1] - indptr[major],
                    bytes_per_source_buffer_element);
            }
            if (selected_buffer_bytes <= bounded_options.max_buffer_bytes ||
                batch_end == batch_begin + 1) {
                break;
            }
            batch_end = batch_begin + std::max<std::size_t>(
                1, (batch_end - batch_begin) / 2);
        }

        auto batch_spans = build_compressed_spans(
            indptr, unique,
            bytes_per_source_buffer_element, bytes_per_source_element,
            physical_chunk_elements, bounded_options, false);
        std::unordered_map<std::uint64_t, CachedMajor> cache;
        cache.reserve(unique.size());
        std::uint64_t cached_bytes = 0;
        for (const auto& span : batch_spans) {
            std::vector<unsigned char> source_data;
            std::vector<unsigned char> source_indices_raw;
            std::vector<std::uint64_t> source_indices;
            read_span(
                span, source_data, source_indices_raw, source_indices);
            const auto cache_started = Clock::now();
            for (std::size_t position = span.selected_begin;
                 position < span.selected_end; ++position) {
                const auto major = unique[position];
                const auto local_start =
                    indptr[major] - span.element_start;
                const auto count = indptr[major + 1] - indptr[major];
                auto& entry = cache[major];
                entry.data.resize(static_cast<std::size_t>(
                    checked_bytes(count, data_item_size)));
                if (!entry.data.empty()) {
                    std::memcpy(
                        entry.data.data(),
                        source_data.data() +
                            checked_bytes(local_start, data_item_size),
                        entry.data.size());
                }
                entry.indices.assign(
                    source_indices.begin() +
                        static_cast<std::ptrdiff_t>(local_start),
                    source_indices.begin() +
                        static_cast<std::ptrdiff_t>(local_start + count));
                cached_bytes += entry.data.size() +
                    checked_bytes(entry.indices.size(), sizeof(std::uint64_t));
            }
            const double cache_seconds = elapsed_seconds(cache_started);
            stats.packing_seconds += cache_seconds;
            if (options.collect_span_stats && !stats.spans.empty()) {
                stats.spans.back().packing_seconds += cache_seconds;
            }
            const std::uint64_t source_buffers =
                source_data.size() + source_indices_raw.size() +
                checked_bytes(source_indices.size(), sizeof(std::uint64_t));
            stats.peak_buffer_bytes = std::max(
                stats.peak_buffer_bytes, source_buffers + cached_bytes);
        }
        const auto pack_started = Clock::now();
        for (std::size_t position = batch_begin;
             position < batch_end; ++position) {
            const auto major = major_selection[position];
            const auto& entry = cache.at(major);
            append_major(
                major, indptr[major], entry.data, {}, entry.indices);
            const std::uint64_t current_buffers =
                output_data.size() + output_indices_raw.size() +
                checked_bytes(output_indices_converted.size(), sizeof(long long));
            stats.peak_buffer_bytes = std::max(
                stats.peak_buffer_bytes, cached_bytes + current_buffers);
        }
        stats.packing_seconds += elapsed_seconds(pack_started);
        batch_begin = batch_end;
    }
    flush_buffers();

    if (output_indptr.size() != major_selection.size() + 1) {
        throw std::runtime_error("Internal H5AD transfer indptr length mismatch");
    }
    const bool output_indptr_int32 =
        written_nnz <=
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    std::vector<long long> signed_indptr(output_indptr.size(), 0);
    for (std::size_t i = 0; i < output_indptr.size(); ++i) {
        if (output_indptr[i] >
            static_cast<std::uint64_t>(std::numeric_limits<long long>::max())) {
            throw std::runtime_error("Destination H5AD sparse indptr exceeds int64");
        }
        signed_indptr[i] = static_cast<long long>(output_indptr[i]);
    }
    const hsize_t indptr_dims[1] = {
        static_cast<hsize_t>(signed_indptr.size())};
    H5Space indptr_space(H5Screate_simple(1, indptr_dims, nullptr));
    H5Property indptr_dcpl = dataset_dcpl(
        source.indptr.get(), {indptr_dims[0]}, false, options.layout_policy);
    H5Dataset destination_indptr(H5Dcreate2(
        destination_group.get(), "indptr",
        output_indptr_int32 ? H5T_STD_I32LE : H5T_STD_I64LE,
        indptr_space.get(), H5P_DEFAULT, indptr_dcpl.get(), H5P_DEFAULT));
    check_h5(static_cast<bool>(destination_indptr),
             "Failed to create destination H5AD indptr");
    const auto indptr_write_started = Clock::now();
    if (!signed_indptr.empty()) {
        check_h5(H5Dwrite(destination_indptr.get(), H5T_NATIVE_LLONG,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT,
                          signed_indptr.data()) >= 0,
                 "Failed to write destination H5AD indptr");
        ++stats.hdf5_write_calls;
    }
    stats.destination_write_seconds += elapsed_seconds(indptr_write_started);

    const auto flush_started = Clock::now();
    check_h5(H5Fflush(destination_file, H5F_SCOPE_LOCAL) >= 0,
             "Failed to flush destination HDF5 file");
    stats.flush_seconds = elapsed_seconds(flush_started);
    // Drain any remaining dirty output pages to disk now, while the transfer
    // still owns the file, instead of deferring the whole tail to the Python
    // commit fsync. Best-effort; final durability is still enforced by the
    // caller before atomic publication.
    const auto final_fsync_started = Clock::now();
    flush_and_fsync_file(destination_file, false);
    stats.destination_fsync_seconds += elapsed_seconds(final_fsync_started);
    stats.destination_bytes_written =
        checked_bytes(written_nnz, source.info.data_item_size) +
        checked_bytes(written_nnz,
                      output_indices_int32 ? sizeof(std::int32_t)
                                           : sizeof(std::int64_t)) +
        checked_bytes(output_indptr.size(),
                      output_indptr_int32 ? sizeof(std::int32_t)
                                          : sizeof(std::int64_t));
    stats.gap_bytes_read =
        stats.source_bytes_read > stats.selected_source_bytes
        ? stats.source_bytes_read - stats.selected_source_bytes : 0;
    stats.destination = source.info;
    stats.destination.rows = output_rows;
    stats.destination.cols = output_cols;
    stats.destination.nnz = written_nnz;
    stats.destination.indices_item_size =
        output_indices_int32 ? sizeof(std::int32_t) : sizeof(std::int64_t);
    stats.destination.indptr_item_size =
        output_indptr_int32 ? sizeof(std::int32_t) : sizeof(std::int64_t);
    stats.destination.encoding_type = is_csr ? "csr_matrix" : "csc_matrix";
    stats.destination.encoding_version = "0.1.0";
    stats.destination.chunked = inspect_layout(destination_data.get()).chunked ||
                                inspect_layout(destination_indices.get()).chunked ||
                                inspect_layout(destination_indptr.get()).chunked;
    stats.destination.filtered = inspect_layout(destination_data.get()).filtered ||
                                 inspect_layout(destination_indices.get()).filtered ||
                                 inspect_layout(destination_indptr.get()).filtered;
    reset_dataset_inventory(stats.destination);
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination_data.get(), "data"));
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination_indices.get(), "indices"));
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination_indptr.get(), "indptr"));
    return stats;
}

struct DenseRowSpan {
    std::uint64_t row_start = 0;
    std::uint64_t row_end = 0;
    std::size_t selected_begin = 0;
    std::size_t selected_end = 0;
};

std::vector<DenseRowSpan> build_dense_spans(
    const std::vector<std::uint64_t>& selected,
    std::size_t row_bytes,
    std::size_t chunk_rows,
    const actionet::h5ad::TransferOptions& options,
    bool force_scan) {
    std::vector<DenseRowSpan> base;
    if (selected.empty()) {
        return base;
    }
    const std::size_t max_rows_by_bytes =
        std::max<std::size_t>(1, options.max_buffer_bytes /
                                  std::max<std::size_t>(1, row_bytes));
    const std::size_t max_rows =
        std::max<std::size_t>(1, std::min(max_rows_by_bytes,
                                         options.max_rows_per_batch));
    std::size_t begin = 0;
    while (begin < selected.size()) {
        std::size_t end = begin + 1;
        while (end < selected.size() &&
               selected[end] == selected[end - 1] + 1 &&
               end - begin < max_rows) {
            ++end;
        }
        base.push_back(DenseRowSpan{
            selected[begin], selected[end - 1] + 1, begin, end});
        begin = end;
    }
    std::vector<DenseRowSpan> out;
    for (const auto& next : base) {
        if (out.empty()) {
            out.push_back(next);
            continue;
        }
        auto& current = out.back();
        const std::uint64_t combined_rows = next.row_end - current.row_start;
        const std::uint64_t gap_rows = next.row_start - current.row_end;
        const bool same_chunk =
            chunk_rows > 0 && current.row_end > current.row_start &&
            (current.row_end - 1) / chunk_rows == next.row_start / chunk_rows;
        const bool selected_limit =
            next.selected_end - current.selected_begin <= max_rows;
        if (selected_limit && combined_rows <= max_rows &&
            (force_scan || checked_bytes(gap_rows, row_bytes) <=
                               options.gap_merge_bytes || same_chunk)) {
            current.row_end = next.row_end;
            current.selected_end = next.selected_end;
        } else {
            out.push_back(next);
        }
    }
    return out;
}

void read_dense_rows(hid_t dataset, hid_t mem_type,
                     std::uint64_t row_start, std::uint64_t row_count,
                     std::uint64_t cols, void* destination) {
    if (row_count == 0 || cols == 0) {
        return;
    }
    const hsize_t starts[2] = {
        static_cast<hsize_t>(row_start), 0};
    const hsize_t counts[2] = {
        static_cast<hsize_t>(row_count), static_cast<hsize_t>(cols)};
    H5Space file_space(H5Dget_space(dataset));
    check_h5(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET,
                                 starts, nullptr, counts, nullptr) >= 0,
             "Failed to select dense source rows");
    H5Space mem_space(H5Screate_simple(2, counts, nullptr));
    check_h5(H5Dread(dataset, mem_type, mem_space.get(), file_space.get(),
                     H5P_DEFAULT, destination) >= 0,
             "Failed to read dense source rows");
}

void write_dense_rows(hid_t dataset, hid_t mem_type,
                      std::uint64_t output_row_start,
                      std::uint64_t row_count,
                      std::uint64_t cols,
                      const void* source) {
    if (row_count == 0 || cols == 0) {
        return;
    }
    const hsize_t starts[2] = {
        static_cast<hsize_t>(output_row_start), 0};
    const hsize_t counts[2] = {
        static_cast<hsize_t>(row_count), static_cast<hsize_t>(cols)};
    H5Space file_space(H5Dget_space(dataset));
    check_h5(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET,
                                 starts, nullptr, counts, nullptr) >= 0,
             "Failed to select dense destination rows");
    H5Space mem_space(H5Screate_simple(2, counts, nullptr));
    check_h5(H5Dwrite(dataset, mem_type, mem_space.get(), file_space.get(),
                      H5P_DEFAULT, source) >= 0,
             "Failed to write dense destination rows");
}

actionet::h5ad::TransferStats transfer_dense(
    const OpenMatrix& source,
    hid_t destination_file,
    const std::string& destination_path,
    const actionet::h5ad::AxisSelection& row_axis_selection,
    const actionet::h5ad::AxisSelection& column_axis_selection,
    const actionet::h5ad::TransferOptions& options) {
    using actionet::h5ad::TransferStats;

    const auto planning_started = Clock::now();
    const auto rows = resolve_axis(row_axis_selection, source.info.rows);
    const auto cols = resolve_axis(column_axis_selection, source.info.cols);
    const bool ordered_unique = rows.size() < 2 || strictly_increasing(rows);
    const bool columns_identity = is_identity(column_axis_selection, source.info.cols);
    const std::size_t item_size = source.info.data_item_size;
    const std::size_t source_row_bytes = static_cast<std::size_t>(
        checked_bytes(source.info.cols, item_size));
    const std::size_t output_row_bytes = static_cast<std::size_t>(
        checked_bytes(cols.size(), item_size));
    if (source_row_bytes >
        std::numeric_limits<std::size_t>::max() - output_row_bytes) {
        throw std::runtime_error("Dense H5AD row buffer size overflow");
    }
    const std::size_t row_buffer_bytes =
        source_row_bytes + output_row_bytes;
    if (row_buffer_bytes > options.max_buffer_bytes) {
        throw std::runtime_error(
            "One dense H5AD row exceeds the configured transfer buffer");
    }
    const auto layout = inspect_layout(source.object.get());
    const std::size_t chunk_rows =
        layout.chunked && !layout.chunks.empty()
        ? static_cast<std::size_t>(layout.chunks[0]) : 0;
    std::vector<DenseRowSpan> spans;
    if (ordered_unique) {
        auto gather = build_dense_spans(
            rows, row_buffer_bytes, chunk_rows, options, false);
        auto scan = build_dense_spans(
            rows, row_buffer_bytes, chunk_rows, options, true);
        auto dense_cost = [&](const std::vector<DenseRowSpan>& candidates) {
            std::uint64_t bytes = 0;
            for (const auto& span : candidates) {
                bytes += checked_bytes(span.row_end - span.row_start,
                                       source_row_bytes);
            }
            return bytes + checked_bytes(
                candidates.size(), options.gap_merge_bytes);
        };
        spans = dense_cost(scan) < dense_cost(gather)
            ? std::move(scan) : std::move(gather);
    }

    delete_if_exists(destination_file, destination_path);
    const auto [parent_path, object_name] = split_parent_name(destination_path);
    H5Group parent = ensure_group(destination_file, parent_path);
    H5Type source_type(H5Dget_type(source.object.get()));
    const hsize_t dims[2] = {
        static_cast<hsize_t>(rows.size()),
        static_cast<hsize_t>(cols.size())};
    H5Space output_space(H5Screate_simple(2, dims, nullptr));
    H5Property dcpl = dataset_dcpl(
        source.object.get(), {dims[0], dims[1]}, false, options.layout_policy);
    H5Dataset destination(H5Dcreate2(
        parent.get(), object_name.c_str(), source_type.get(), output_space.get(),
        H5P_DEFAULT, dcpl.get(), H5P_DEFAULT));
    check_h5(static_cast<bool>(destination),
             "Failed to create dense destination H5AD matrix");
    write_string_attr(destination.get(), "encoding-type", "array");
    write_string_attr(destination.get(), "encoding-version", "0.2.0");

    TransferStats stats;
    stats.source = source.info;
    stats.selected_source_bytes = cols.empty()
        ? 0 : checked_bytes(rows.size(), source_row_bytes);
    stats.planning_seconds = elapsed_seconds(planning_started);
    stats.span_count = 0;
    std::uint64_t output_row = 0;
    std::uint64_t dense_bytes_since_fsync = 0;
    auto dense_periodic_fsync = [&](std::uint64_t just_written_rows) {
        dense_bytes_since_fsync +=
            checked_bytes(just_written_rows, output_row_bytes);
        if (dense_bytes_since_fsync >= kIncrementalFsyncBytes) {
            const auto fsync_started = Clock::now();
            flush_and_fsync_file(destination_file, true);
            stats.destination_fsync_seconds += elapsed_seconds(fsync_started);
            dense_bytes_since_fsync = 0;
        }
    };
    auto pack_row = [&](unsigned char* output,
                        const unsigned char* input) {
        if (columns_identity) {
            std::memcpy(output, input, output_row_bytes);
            return;
        }
        std::size_t offset = 0;
        for (const auto col : cols) {
            std::memcpy(
                output + offset,
                input + checked_bytes(col, item_size),
                item_size);
            offset += item_size;
        }
    };

    if (!cols.empty() && ordered_unique) {
        for (const auto& span : spans) {
            const std::uint64_t source_row_count =
                span.row_end - span.row_start;
            const std::uint64_t source_elements =
                source_row_count * source.info.cols;
            std::vector<unsigned char> source_buffer(
                static_cast<std::size_t>(
                    checked_bytes(source_elements, item_size)));
            const auto read_started = Clock::now();
            read_dense_rows(
                source.object.get(), source_type.get(),
                span.row_start, source_row_count,
                source.info.cols, source_buffer.data());
            const double source_read_seconds =
                elapsed_seconds(read_started);
            stats.source_read_seconds += source_read_seconds;
            ++stats.hdf5_read_calls;
            ++stats.span_count;
            stats.source_bytes_read +=
                checked_bytes(source_elements, item_size);

            const std::size_t selected_rows =
                span.selected_end - span.selected_begin;
            std::vector<unsigned char> output_buffer(
                static_cast<std::size_t>(
                    checked_bytes(selected_rows, output_row_bytes)));
            const auto pack_started = Clock::now();
            for (std::size_t selected_position = span.selected_begin;
                 selected_position < span.selected_end; ++selected_position) {
                const std::uint64_t local_row =
                    rows[selected_position] - span.row_start;
                pack_row(
                    output_buffer.data() +
                        (selected_position - span.selected_begin) *
                            output_row_bytes,
                    source_buffer.data() +
                        checked_bytes(
                            local_row * source.info.cols, item_size));
            }
            const double packing_seconds =
                elapsed_seconds(pack_started);
            stats.packing_seconds += packing_seconds;
            stats.peak_buffer_bytes = std::max<std::uint64_t>(
                stats.peak_buffer_bytes,
                source_buffer.size() + output_buffer.size());
            const auto write_started = Clock::now();
            write_dense_rows(
                destination.get(), source_type.get(), output_row,
                selected_rows, cols.size(), output_buffer.data());
            stats.destination_write_seconds += elapsed_seconds(write_started);
            ++stats.hdf5_write_calls;
            output_row += selected_rows;
            dense_periodic_fsync(selected_rows);
            if (options.collect_span_stats) {
                stats.spans.push_back(actionet::h5ad::SpanStats{
                    span.row_start,
                    span.row_end,
                    selected_rows,
                    source_elements,
                    checked_bytes(source_elements, item_size),
                    source_read_seconds,
                    packing_seconds,
                });
            }
        }
    } else if (!cols.empty()) {
        const std::size_t max_batch_rows = std::max<std::size_t>(
            1, std::min<std::size_t>(
                   options.max_rows_per_batch,
                   options.max_buffer_bytes /
                       std::max<std::size_t>(1, row_buffer_bytes)));
        for (std::size_t batch_begin = 0;
             batch_begin < rows.size();
             batch_begin += max_batch_rows) {
            const std::size_t batch_end = std::min(
                rows.size(), batch_begin + max_batch_rows);
            const std::size_t batch_rows = batch_end - batch_begin;
            std::vector<unsigned char> output_buffer(
                static_cast<std::size_t>(
                    checked_bytes(batch_rows, output_row_bytes)));

            std::unordered_map<std::uint64_t, std::vector<std::size_t>>
                output_positions;
            output_positions.reserve(batch_rows);
            std::vector<std::uint64_t> unique_rows;
            unique_rows.reserve(batch_rows);
            for (std::size_t position = batch_begin;
                 position < batch_end; ++position) {
                auto& positions = output_positions[rows[position]];
                if (positions.empty()) {
                    unique_rows.push_back(rows[position]);
                }
                positions.push_back(position - batch_begin);
            }
            std::sort(unique_rows.begin(), unique_rows.end());

            actionet::h5ad::TransferOptions span_options = options;
            span_options.max_buffer_bytes = std::max<std::size_t>(
                1, options.max_buffer_bytes - output_buffer.size());
            auto gather = build_dense_spans(
                unique_rows, source_row_bytes, chunk_rows,
                span_options, false);
            auto scan = build_dense_spans(
                unique_rows, source_row_bytes, chunk_rows,
                span_options, true);
            auto dense_cost = [&](const std::vector<DenseRowSpan>& candidates) {
                std::uint64_t bytes = 0;
                for (const auto& span : candidates) {
                    bytes += checked_bytes(
                        span.row_end - span.row_start, source_row_bytes);
                }
                return bytes + checked_bytes(
                    candidates.size(), options.gap_merge_bytes);
            };
            auto batch_spans = dense_cost(scan) < dense_cost(gather)
                ? std::move(scan) : std::move(gather);

            for (const auto& span : batch_spans) {
                const std::uint64_t source_row_count =
                    span.row_end - span.row_start;
                const std::uint64_t source_elements =
                    source_row_count * source.info.cols;
                std::vector<unsigned char> source_buffer(
                    static_cast<std::size_t>(
                        checked_bytes(source_elements, item_size)));
                const auto read_started = Clock::now();
                read_dense_rows(
                    source.object.get(), source_type.get(),
                    span.row_start, source_row_count,
                    source.info.cols, source_buffer.data());
                const double source_read_seconds =
                    elapsed_seconds(read_started);
                stats.source_read_seconds += source_read_seconds;
                ++stats.hdf5_read_calls;
                ++stats.span_count;
                stats.source_bytes_read +=
                    checked_bytes(source_elements, item_size);

                const auto pack_started = Clock::now();
                for (std::size_t selected_position = span.selected_begin;
                     selected_position < span.selected_end;
                     ++selected_position) {
                    const auto source_row = unique_rows[selected_position];
                    const auto* input =
                        source_buffer.data() +
                        checked_bytes(
                            (source_row - span.row_start) * source.info.cols,
                            item_size);
                    for (const auto output_position :
                         output_positions.at(source_row)) {
                        pack_row(
                            output_buffer.data() +
                                output_position * output_row_bytes,
                            input);
                    }
                }
                const double packing_seconds =
                    elapsed_seconds(pack_started);
                stats.packing_seconds += packing_seconds;
                stats.peak_buffer_bytes = std::max<std::uint64_t>(
                    stats.peak_buffer_bytes,
                    source_buffer.size() + output_buffer.size());
                if (options.collect_span_stats) {
                    stats.spans.push_back(actionet::h5ad::SpanStats{
                        span.row_start,
                        span.row_end,
                        static_cast<std::uint64_t>(
                            span.selected_end - span.selected_begin),
                        source_elements,
                        checked_bytes(source_elements, item_size),
                        source_read_seconds,
                        packing_seconds,
                    });
                }
            }
            const auto write_started = Clock::now();
            write_dense_rows(
                destination.get(), source_type.get(), output_row,
                batch_rows, cols.size(), output_buffer.data());
            stats.destination_write_seconds += elapsed_seconds(write_started);
            ++stats.hdf5_write_calls;
            output_row += batch_rows;
            dense_periodic_fsync(batch_rows);
        }
    }

    const auto flush_started = Clock::now();
    check_h5(H5Fflush(destination_file, H5F_SCOPE_LOCAL) >= 0,
             "Failed to flush dense destination HDF5 matrix");
    stats.flush_seconds = elapsed_seconds(flush_started);
    const auto dense_final_fsync_started = Clock::now();
    flush_and_fsync_file(destination_file, false);
    stats.destination_fsync_seconds += elapsed_seconds(dense_final_fsync_started);
    const std::uint64_t output_elements = checked_product(
        static_cast<hsize_t>(rows.size()),
        static_cast<hsize_t>(cols.size()),
        "Dense destination element count");
    stats.destination_bytes_written =
        checked_bytes(output_elements, item_size);
    stats.gap_bytes_read =
        stats.source_bytes_read > stats.selected_source_bytes
        ? stats.source_bytes_read - stats.selected_source_bytes : 0;
    stats.destination = source.info;
    stats.destination.rows = rows.size();
    stats.destination.cols = cols.size();
    stats.destination.nnz = output_elements;
    stats.destination.chunked = inspect_layout(destination.get()).chunked;
    stats.destination.filtered = inspect_layout(destination.get()).filtered;
    reset_dataset_inventory(stats.destination);
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination.get(), "data"));
    return stats;
}

double transform_value(
    double value,
    double scale,
    const actionet::h5ad::TransformOptions& options) {
    double output = value * scale;
    if (options.apply_log) {
        output = std::log(output + options.pseudocount) * options.log_scale;
    }
    return output;
}

actionet::h5ad::TransferStats transform_dense(
    const OpenMatrix& source,
    hid_t destination_file,
    const std::string& destination_path,
    const actionet::h5ad::TransformOptions& options) {
    using actionet::h5ad::TransferStats;

    delete_if_exists(destination_file, destination_path);
    const auto [parent_path, object_name] = split_parent_name(destination_path);
    H5Group parent = ensure_group(destination_file, parent_path);
    const hsize_t dims[2] = {
        static_cast<hsize_t>(source.info.rows),
        static_cast<hsize_t>(source.info.cols)};
    H5Space space(H5Screate_simple(2, dims, nullptr));
    H5Property dcpl = dataset_dcpl(
        source.object.get(), {dims[0], dims[1]}, false,
        options.transfer.layout_policy);
    const hid_t output_type =
        options.output_dtype == actionet::h5ad::TransformDType::Float32
        ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
    H5Dataset destination(H5Dcreate2(
        parent.get(), object_name.c_str(), output_type, space.get(),
        H5P_DEFAULT, dcpl.get(), H5P_DEFAULT));
    check_h5(static_cast<bool>(destination),
             "Failed to create transformed dense H5AD matrix");
    write_string_attr(destination.get(), "encoding-type", "array");
    write_string_attr(destination.get(), "encoding-version", "0.2.0");

    TransferStats stats;
    stats.source = source.info;
    stats.selected_source_bytes = checked_bytes(
        source.info.rows * source.info.cols, source.info.data_item_size);
    const std::size_t bytes_per_row =
        static_cast<std::size_t>(checked_bytes(
            source.info.cols, sizeof(double) * 2));
    const std::size_t rows_by_memory = std::max<std::size_t>(
        1, options.transfer.max_buffer_bytes /
               std::max<std::size_t>(1, bytes_per_row));
    const std::size_t row_batch = std::max<std::size_t>(
        1, std::min(rows_by_memory, options.transfer.max_rows_per_batch));

    const std::size_t transform_dense_output_item_size =
        options.output_dtype == actionet::h5ad::TransformDType::Float32 ? 4 : 8;
    const std::size_t transform_dense_row_bytes = static_cast<std::size_t>(
        checked_bytes(source.info.cols, transform_dense_output_item_size));
    std::uint64_t transform_dense_bytes_since_fsync = 0;

    for (std::uint64_t row_start = 0;
         row_start < source.info.rows;
         row_start += row_batch) {
        const std::uint64_t row_count = std::min<std::uint64_t>(
            row_batch, source.info.rows - row_start);
        const std::uint64_t element_count = row_count * source.info.cols;
        std::vector<double> values(static_cast<std::size_t>(element_count), 0.0);
        const auto read_started = Clock::now();
        read_dense_rows(
            source.object.get(), H5T_NATIVE_DOUBLE, row_start, row_count,
            source.info.cols, values.data());
        stats.source_read_seconds += elapsed_seconds(read_started);
        ++stats.hdf5_read_calls;
        stats.source_bytes_read += checked_bytes(
            element_count, source.info.data_item_size);

        const auto packing_started = Clock::now();
        for (std::uint64_t local_row = 0; local_row < row_count; ++local_row) {
            const double scale =
                options.row_scale[static_cast<std::size_t>(row_start + local_row)];
            const std::uint64_t offset = local_row * source.info.cols;
            for (std::uint64_t col = 0; col < source.info.cols; ++col) {
                values[static_cast<std::size_t>(offset + col)] = transform_value(
                    values[static_cast<std::size_t>(offset + col)], scale, options);
            }
        }
        stats.packing_seconds += elapsed_seconds(packing_started);
        stats.peak_buffer_bytes = std::max<std::uint64_t>(
            stats.peak_buffer_bytes,
            checked_bytes(values.size(), sizeof(double)));

        const auto write_started = Clock::now();
        // The in-memory transform buffer is always double, but the destination
        // dataset dtype is authoritative (H5T_IEEE_F32LE under Float32). Passing
        // H5T_NATIVE_DOUBLE as the memory type lets HDF5 narrow double->float32
        // per element at write time. This single narrowing point is intentional;
        // do not "fix" it by widening the on-disk dtype.
        write_dense_rows(
            destination.get(), H5T_NATIVE_DOUBLE, row_start, row_count,
            source.info.cols, values.data());
        stats.destination_write_seconds += elapsed_seconds(write_started);
        ++stats.hdf5_write_calls;
        ++stats.span_count;
        transform_dense_bytes_since_fsync +=
            checked_bytes(row_count, transform_dense_row_bytes);
        if (transform_dense_bytes_since_fsync >= kIncrementalFsyncBytes) {
            const auto fsync_started = Clock::now();
            flush_and_fsync_file(destination_file, true);
            stats.destination_fsync_seconds += elapsed_seconds(fsync_started);
            transform_dense_bytes_since_fsync = 0;
        }
    }

    const auto flush_started = Clock::now();
    check_h5(H5Fflush(destination_file, H5F_SCOPE_LOCAL) >= 0,
             "Failed to flush transformed dense H5AD matrix");
    stats.flush_seconds = elapsed_seconds(flush_started);
    const auto transform_dense_final_fsync = Clock::now();
    flush_and_fsync_file(destination_file, false);
    stats.destination_fsync_seconds += elapsed_seconds(transform_dense_final_fsync);
    const std::size_t output_item_size =
        options.output_dtype == actionet::h5ad::TransformDType::Float32 ? 4 : 8;
    stats.destination_bytes_written = checked_bytes(
        source.info.rows * source.info.cols, output_item_size);
    stats.destination = source.info;
    stats.destination.data_item_size = output_item_size;
    // Verify the dtype we actually created on disk (H5T_IEEE_F32LE / F64LE)
    // matches the recorded width, independent of the double working buffer
    // that HDF5 narrows at write time.
    {
        H5Type destination_type(H5Dget_type(destination.get()));
        check_h5(static_cast<bool>(destination_type),
                 "Failed to inspect transformed dense destination dtype");
        check_h5(H5Tget_size(destination_type.get()) == output_item_size,
                 "Transformed dense destination dtype width is inconsistent");
    }
    stats.destination.chunked = inspect_layout(destination.get()).chunked;
    stats.destination.filtered = inspect_layout(destination.get()).filtered;
    reset_dataset_inventory(stats.destination);
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination.get(), "data"));
    return stats;
}

actionet::h5ad::TransferStats transform_compressed(
    const OpenMatrix& source,
    hid_t destination_file,
    const std::string& destination_path,
    const actionet::h5ad::TransformOptions& options) {
    using actionet::h5ad::MatrixEncoding;
    using actionet::h5ad::TransferStats;

    const bool is_csr = source.info.encoding == MatrixEncoding::CSR;
    const std::uint64_t minor_size =
        is_csr ? source.info.cols : source.info.rows;
    const auto indptr = read_integer_1d(source.indptr.get());
    bool hard_link_structure = options.destination_structure_path.has_value();
    std::string structure_path;
    if (hard_link_structure) {
        structure_path = normalize_h5_path(*options.destination_structure_path);
        if (structure_path == normalize_h5_path(destination_path)) {
            throw std::invalid_argument(
                "Sparse transform structure path must differ from destination");
        }
        OpenMatrix structure =
            open_matrix(destination_file, structure_path, true);
        if (structure.info.encoding != source.info.encoding ||
            structure.info.rows != source.info.rows ||
            structure.info.cols != source.info.cols ||
            structure.info.nnz != source.info.nnz) {
            throw std::invalid_argument(
                "Sparse transform hard-link source has incompatible structure");
        }
        if (!options.destination_structure_is_exact_copy) {
            if (read_integer_1d(structure.indptr.get()) != indptr) {
                throw std::invalid_argument(
                    "Sparse transform hard-link source has different indptr values");
            }
            const std::size_t compare_batch = std::max<std::size_t>(
                1, options.transfer.max_buffer_bytes /
                       (2 * sizeof(std::uint64_t)));
            for (std::uint64_t start = 0; start < source.info.nnz;
                 start += compare_batch) {
                const std::uint64_t count = std::min<std::uint64_t>(
                    compare_batch, source.info.nnz - start);
                if (read_1d_indices(source.indices.get(), start, count) !=
                    read_1d_indices(structure.indices.get(), start, count)) {
                    throw std::invalid_argument(
                        "Sparse transform hard-link source has different indices");
                }
            }
        }
    }

    delete_if_exists(destination_file, destination_path);
    const auto [parent_path, object_name] = split_parent_name(destination_path);
    H5Group parent = ensure_group(destination_file, parent_path);
    H5Group destination_group(H5Gcreate2(
        parent.get(), object_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    check_h5(static_cast<bool>(destination_group),
             "Failed to create transformed sparse H5AD group");
    write_string_attr(
        destination_group.get(), "encoding-type",
        is_csr ? "csr_matrix" : "csc_matrix");
    write_string_attr(destination_group.get(), "encoding-version", "0.1.0");
    write_shape_attr(
        destination_group.get(), source.info.rows, source.info.cols);

    const hid_t output_data_type =
        options.output_dtype == actionet::h5ad::TransformDType::Float32
        ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
    H5Dataset destination_data = create_1d_dataset(
        destination_group.get(), "data", output_data_type, source.data.get(),
        source.info.nnz, false, options.transfer.layout_policy);

    TransferStats stats;
    stats.source = source.info;
    stats.selected_source_bytes =
        checked_bytes(source.info.nnz, source.info.data_item_size);

    if (hard_link_structure) {
        const std::string existing = structure_path + "/indices";
        check_h5(H5Lcreate_hard(
                     destination_file, existing.c_str(),
                     destination_group.get(), "indices",
                     H5P_DEFAULT, H5P_DEFAULT) >= 0,
                 "Failed to hard-link transformed sparse indices");
    } else {
        const bool indices_int32 =
            minor_size <= static_cast<std::uint64_t>(
                              std::numeric_limits<std::int32_t>::max());
        H5Dataset destination_indices = create_1d_dataset(
            destination_group.get(), "indices",
            indices_int32 ? H5T_STD_I32LE : H5T_STD_I64LE,
            source.indices.get(), source.info.nnz, false,
            options.transfer.layout_policy);
        const std::size_t index_batch = std::max<std::size_t>(
            1, options.transfer.max_buffer_bytes / (sizeof(std::uint64_t) * 2));
        for (std::uint64_t start = 0; start < source.info.nnz;
             start += index_batch) {
            const std::uint64_t count = std::min<std::uint64_t>(
                index_batch, source.info.nnz - start);
            const auto read_started = Clock::now();
            const auto indices =
                read_1d_indices(source.indices.get(), start, count);
            stats.source_read_seconds += elapsed_seconds(read_started);
            ++stats.hdf5_read_calls;
            stats.source_bytes_read += checked_bytes(
                count, source.info.indices_item_size);
            std::vector<long long> signed_indices(indices.begin(), indices.end());
            const auto write_started = Clock::now();
            write_1d(
                destination_indices.get(), H5T_NATIVE_LLONG,
                start, count, signed_indices.data(), false);
            stats.destination_write_seconds += elapsed_seconds(write_started);
            ++stats.hdf5_write_calls;
        }

    }

    // Keep indptr independent even when indices are hard-linked. Structural
    // rewrites of either matrix must not mutate the other layer's row/column
    // pointer object.
    const bool indptr_int32 =
        source.info.nnz <= static_cast<std::uint64_t>(
                               std::numeric_limits<std::int32_t>::max());
    const hsize_t indptr_dims[1] = {static_cast<hsize_t>(indptr.size())};
    H5Space indptr_space(H5Screate_simple(1, indptr_dims, nullptr));
    H5Property indptr_dcpl = dataset_dcpl(
        source.indptr.get(), {indptr_dims[0]}, false,
        options.transfer.layout_policy);
    H5Dataset destination_indptr(H5Dcreate2(
        destination_group.get(), "indptr",
        indptr_int32 ? H5T_STD_I32LE : H5T_STD_I64LE,
        indptr_space.get(), H5P_DEFAULT, indptr_dcpl.get(), H5P_DEFAULT));
    std::vector<long long> signed_indptr(indptr.begin(), indptr.end());
    const auto indptr_write_started = Clock::now();
    if (!signed_indptr.empty()) {
        check_h5(H5Dwrite(
                     destination_indptr.get(), H5T_NATIVE_LLONG,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     signed_indptr.data()) >= 0,
                 "Failed to write transformed sparse indptr");
        ++stats.hdf5_write_calls;
    }
    stats.destination_write_seconds += elapsed_seconds(indptr_write_started);
    stats.source_bytes_read += checked_bytes(
        indptr.size(), source.info.indptr_item_size);

    const std::size_t working_bytes_per_element =
        sizeof(double) + (is_csr ? 0 : sizeof(std::uint64_t));
    const std::size_t batch_elements = std::max<std::size_t>(
        1, options.transfer.max_buffer_bytes /
               std::max<std::size_t>(1, working_bytes_per_element));
    std::uint64_t csr_row = 0;
    const std::size_t transform_sparse_output_item_size =
        options.output_dtype == actionet::h5ad::TransformDType::Float32 ? 4 : 8;
    std::uint64_t transform_sparse_bytes_since_fsync = 0;
    for (std::uint64_t start = 0; start < source.info.nnz;
         start += batch_elements) {
        const std::uint64_t count = std::min<std::uint64_t>(
            batch_elements, source.info.nnz - start);
        std::vector<double> values(static_cast<std::size_t>(count), 0.0);
        std::vector<std::uint64_t> row_indices;
        const auto read_started = Clock::now();
        read_1d_raw(
            source.data.get(), H5T_NATIVE_DOUBLE, start, count, values.data());
        ++stats.hdf5_read_calls;
        stats.source_bytes_read += checked_bytes(
            count, source.info.data_item_size);
        if (!is_csr) {
            row_indices = read_1d_indices(source.indices.get(), start, count);
            ++stats.hdf5_read_calls;
            stats.source_bytes_read += checked_bytes(
                count, source.info.indices_item_size);
        }
        stats.source_read_seconds += elapsed_seconds(read_started);

        const auto packing_started = Clock::now();
        for (std::uint64_t local = 0; local < count; ++local) {
            std::uint64_t row = 0;
            if (is_csr) {
                const std::uint64_t position = start + local;
                while (csr_row + 1 < indptr.size() &&
                       position >= indptr[static_cast<std::size_t>(csr_row + 1)]) {
                    ++csr_row;
                }
                row = csr_row;
            } else {
                row = row_indices[static_cast<std::size_t>(local)];
                if (row >= source.info.rows) {
                    throw std::runtime_error(
                        "CSC transform encountered an out-of-bounds row index");
                }
            }
            values[static_cast<std::size_t>(local)] = transform_value(
                values[static_cast<std::size_t>(local)],
                options.row_scale[static_cast<std::size_t>(row)],
                options);
        }
        stats.packing_seconds += elapsed_seconds(packing_started);
        stats.peak_buffer_bytes = std::max<std::uint64_t>(
            stats.peak_buffer_bytes,
            checked_bytes(values.size(), sizeof(double)) +
                checked_bytes(row_indices.size(), sizeof(std::uint64_t)));

        const auto write_started = Clock::now();
        // As in transform_dense: the working buffer is double, the destination
        // "data" dataset dtype is authoritative (F32LE under Float32), and HDF5
        // narrows double->float32 per element here. Intentional single point of
        // narrowing; do not widen the on-disk dtype.
        write_1d(
            destination_data.get(), H5T_NATIVE_DOUBLE,
            start, count, values.data(), false);
        stats.destination_write_seconds += elapsed_seconds(write_started);
        ++stats.hdf5_write_calls;
        ++stats.span_count;
        transform_sparse_bytes_since_fsync +=
            checked_bytes(count, transform_sparse_output_item_size);
        if (transform_sparse_bytes_since_fsync >= kIncrementalFsyncBytes) {
            const auto fsync_started = Clock::now();
            flush_and_fsync_file(destination_file, true);
            stats.destination_fsync_seconds += elapsed_seconds(fsync_started);
            transform_sparse_bytes_since_fsync = 0;
        }
    }

    const auto flush_started = Clock::now();
    check_h5(H5Fflush(destination_file, H5F_SCOPE_LOCAL) >= 0,
             "Failed to flush transformed sparse H5AD matrix");
    stats.flush_seconds = elapsed_seconds(flush_started);
    const auto transform_sparse_final_fsync = Clock::now();
    flush_and_fsync_file(destination_file, false);
    stats.destination_fsync_seconds += elapsed_seconds(transform_sparse_final_fsync);
    const std::size_t output_item_size =
        options.output_dtype == actionet::h5ad::TransformDType::Float32 ? 4 : 8;
    stats.destination_bytes_written =
        checked_bytes(source.info.nnz, output_item_size);
    stats.destination = source.info;
    stats.destination.data_item_size = output_item_size;
    // Verify the dtype we actually created on disk (H5T_IEEE_F32LE / F64LE)
    // matches the recorded width, independent of the double working buffer
    // that HDF5 narrows at write time.
    {
        H5Type destination_type(H5Dget_type(destination_data.get()));
        check_h5(static_cast<bool>(destination_type),
                 "Failed to inspect transformed sparse destination dtype");
        check_h5(H5Tget_size(destination_type.get()) == output_item_size,
                 "Transformed sparse destination dtype width is inconsistent");
    }
    stats.destination.indices_item_size =
        minor_size <= static_cast<std::uint64_t>(
                          std::numeric_limits<std::int32_t>::max()) ? 4 : 8;
    stats.destination.indptr_item_size =
        source.info.nnz <= static_cast<std::uint64_t>(
                               std::numeric_limits<std::int32_t>::max()) ? 4 : 8;
    H5Dataset destination_indices =
        open_dataset(destination_group.get(), "indices");
    stats.destination.chunked = inspect_layout(destination_data.get()).chunked ||
                                inspect_layout(destination_indices.get()).chunked ||
                                inspect_layout(destination_indptr.get()).chunked;
    stats.destination.filtered = inspect_layout(destination_data.get()).filtered ||
                                 inspect_layout(destination_indices.get()).filtered ||
                                 inspect_layout(destination_indptr.get()).filtered;
    reset_dataset_inventory(stats.destination);
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination_data.get(), "data"));
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination_indices.get(), "indices"));
    append_dataset_info(
        stats.destination, inspect_dataset_info(destination_indptr.get(), "indptr"));
    return stats;
}

} // namespace

namespace actionet::h5ad {

MatrixInfo inspect_matrix(const std::string& file_path,
                          const std::string& group_path) {
    H5File file = open_readonly(file_path, "inspect_matrix");
    return open_matrix(file.get(), group_path, true).info;
}

ValidationReport validate_matrix(const std::string& file_path,
                                 const std::string& group_path,
                                 ValidationLevel level) {
    ValidationReport report;
    try {
        H5File file = open_readonly(file_path, "validate_matrix");
        OpenMatrix matrix = open_matrix(file.get(), group_path, true);
        report.info = matrix.info;
        if (level == ValidationLevel::Full &&
            matrix.info.encoding != MatrixEncoding::Dense) {
            const auto indices = read_integer_1d(matrix.indices.get());
            const std::uint64_t minor =
                matrix.info.encoding == MatrixEncoding::CSR
                ? matrix.info.cols : matrix.info.rows;
            for (const auto index : indices) {
                if (index >= minor) {
                    throw std::runtime_error(
                        "H5AD sparse index exceeds minor-axis bounds");
                }
            }
            report.indices_scanned = indices.size();
        }
        report.valid = true;
    } catch (const std::exception& error) {
        report.valid = false;
        report.error = error.what();
    }
    return report;
}

TransferStats subset_matrix(const std::string& source_file_path,
                            const std::string& source_group_path,
                            const std::string& destination_file_path,
                            const std::string& destination_group_path,
                            const AxisSelection& row_selection,
                            const AxisSelection& column_selection,
                            const TransferOptions& options) {
    if (options.max_buffer_bytes == 0 || options.max_rows_per_batch == 0) {
        throw std::invalid_argument(
            "H5AD transfer buffer and row limits must be greater than zero");
    }
    H5File source_file = open_readonly(source_file_path, "subset_matrix");
    OpenMatrix source = open_matrix(source_file.get(), source_group_path, true);

    // Capability validation is deliberately completed before the destination
    // file is opened or its object path is changed.
    resolve_axis(row_selection, source.info.rows);
    resolve_axis(column_selection, source.info.cols);
    validate_transfer_filters(source, options.layout_policy);

    H5File destination_file =
        open_readwrite(destination_file_path, "subset_matrix");
    if (source.info.encoding == MatrixEncoding::Dense) {
        return transfer_dense(
            source, destination_file.get(), destination_group_path,
            row_selection, column_selection, options);
    }
    if (source.info.encoding == MatrixEncoding::CSR) {
        const auto rows = resolve_axis(row_selection, source.info.rows);
        return transfer_compressed(
            source, destination_file.get(), destination_group_path,
            rows, column_selection, options);
    }
    const auto cols = resolve_axis(column_selection, source.info.cols);
    return transfer_compressed(
        source, destination_file.get(), destination_group_path,
        cols, row_selection, options);
}

TransferStats copy_matrix(const std::string& source_file_path,
                          const std::string& source_group_path,
                          const std::string& destination_file_path,
                          const std::string& destination_group_path,
                          const TransferOptions& options) {
    return subset_matrix(
        source_file_path, source_group_path,
        destination_file_path, destination_group_path,
        AxisSelection::all(), AxisSelection::all(), options);
}

TransferStats transform_matrix(
    const std::string& source_file_path,
    const std::string& source_group_path,
    const std::string& destination_file_path,
    const std::string& destination_group_path,
    const TransformOptions& options) {
    if (options.transfer.max_buffer_bytes == 0 ||
        options.transfer.max_rows_per_batch == 0) {
        throw std::invalid_argument(
            "H5AD transform buffer and row limits must be greater than zero");
    }
    if (!std::isfinite(options.pseudocount) || options.pseudocount <= 0.0) {
        throw std::invalid_argument(
            "H5AD transform pseudocount must be finite and positive");
    }
    if (!std::isfinite(options.log_scale)) {
        throw std::invalid_argument("H5AD transform log scale must be finite");
    }

    H5File source_file = open_readonly(source_file_path, "transform_matrix");
    OpenMatrix source = open_matrix(source_file.get(), source_group_path, true);
    if (options.row_scale.size() != source.info.rows) {
        throw std::invalid_argument(
            "H5AD transform row-scale length does not match matrix rows");
    }
    for (const double scale : options.row_scale) {
        if (!std::isfinite(scale)) {
            throw std::invalid_argument(
                "H5AD transform row scales must be finite");
        }
    }
    validate_transfer_filters(source, options.transfer.layout_policy);

    H5File destination_file =
        open_readwrite(destination_file_path, "transform_matrix");
    if (source.info.encoding == MatrixEncoding::Dense) {
        return transform_dense(
            source, destination_file.get(), destination_group_path, options);
    }
    return transform_compressed(
        source, destination_file.get(), destination_group_path, options);
}

} // namespace actionet::h5ad
