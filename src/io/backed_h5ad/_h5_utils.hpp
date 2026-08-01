// SPDX-License-Identifier: MIT
//
// Internal HDF5 helpers shared across the backed_h5ad translation units.
// This header is intentionally private (kept under src/, not include/) so it
// does not participate in libactionet's public API.

#pragma once

#include <hdf5.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#endif

namespace actionet::detail::h5 {

    /// Move-only owner for one HDF5 identifier.
    template <herr_t (*Closer)(hid_t)>
    class Handle {
    public:
        Handle() = default;
        explicit Handle(hid_t id) : id_(id) {}
        ~Handle() { reset(); }

        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& other) noexcept : id_(other.release()) {}
        Handle& operator=(Handle&& other) noexcept {
            if (this != &other) {
                reset(other.release());
            }
            return *this;
        }

        hid_t get() const { return id_; }
        explicit operator bool() const { return id_ >= 0; }
        hid_t release() {
            const hid_t value = id_;
            id_ = -1;
            return value;
        }
        void reset(hid_t next = -1) {
            if (id_ >= 0) {
                Closer(id_);
            }
            id_ = next;
        }

    private:
        hid_t id_ = -1;
    };

    using File = Handle<H5Fclose>;
    using Group = Handle<H5Gclose>;
    using Dataset = Handle<H5Dclose>;
    using Space = Handle<H5Sclose>;
    using Type = Handle<H5Tclose>;
    using Attribute = Handle<H5Aclose>;
    using Property = Handle<H5Pclose>;

    /// Throw ``std::runtime_error(msg)`` when ``ok`` is false.
    ///
    /// Centralises the tiny "assert-or-throw" pattern that the three
    /// backed_h5ad TUs previously duplicated as file-local ``check_h5``.
    inline void check_h5(bool ok, const char* msg) {
        if (!ok) {
            throw std::runtime_error(msg);
        }
    }

    /// Open an HDF5 file read-only with advisory file locking disabled.
    ///
    /// Advisory locking must be disabled so this reader can coexist with an
    /// h5py/AnnData backed-mode handle that already holds an OS lock on the
    /// same inode (otherwise HDF5 fails with EAGAIN / errno 11 from
    /// H5FD__sec2_lock).  ``H5Pset_file_locking(fapl, 0, 1)`` disables locking
    /// and asks HDF5 to ignore the disable when the platform doesn't support
    /// it (avoiding a hard error on those systems).
    ///
    /// @param file_path      Path to the .h5ad file.
    /// @param err_context    Context prefix included in the exception message
    ///                       when opening fails (e.g. "createBackedOperator").
    /// @returns              A valid HDF5 file id that the caller must close
    ///                       via ``H5Fclose``.
    inline hid_t open_h5_readonly_no_lock(const std::string& file_path,
                                          const char* err_context,
                                          size_t sieve_buffer_bytes = 0) {
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        check_h5(fapl >= 0, "Failed to create file access property list");
        H5Pset_file_locking(fapl, 0, 1);
        if (sieve_buffer_bytes > 0) {
            check_h5(
                H5Pset_sieve_buf_size(fapl, sieve_buffer_bytes) >= 0,
                "Failed to configure HDF5 data sieve buffer");
        }
        hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        if (file_id < 0) {
            throw std::runtime_error(
                std::string(err_context) + ": failed to open h5ad file: " + file_path);
        }
        return file_id;
    }

    /// Flush an HDF5 file's cache to the OS and durably persist it to disk.
    ///
    /// The default HDF5 flush (``H5Fflush``) only pushes the library's own
    /// cache into the OS page cache; it does not force writeback to the
    /// storage device. Large native transfers therefore leave the entire
    /// multi-GB output dirty in the page cache, and a single trailing
    /// ``fsync`` in the Python commit path must drain all of it at once
    /// (observed as a minutes-long "post-write" tail on atlas-scale files).
    ///
    /// This helper pushes the HDF5 cache to the virtual file driver, obtains
    /// the backing POSIX file descriptor via ``H5Fget_vfd_handle`` (valid for
    /// the default sec2/stdio drivers), and issues ``fsync``. When
    /// ``drop_cache`` is set it additionally advises the kernel to drop the
    /// now-clean pages (``POSIX_FADV_DONTNEED``) so repeated periodic calls do
    /// not keep the whole output resident. All steps are best-effort: an
    /// unsupported driver or platform simply falls back to ``H5Fflush``
    /// semantics without raising, because durability of the final published
    /// file is still guaranteed by the caller's own ``fsync`` before
    /// ``os.replace``.
    ///
    /// @param file_id     Open, writable HDF5 file identifier.
    /// @param drop_cache  When true, hint the kernel to evict clean pages.
    /// @returns           true if a POSIX ``fsync`` was issued; false when the
    ///                    platform/driver did not expose a usable descriptor.
    inline bool flush_and_fsync_file(hid_t file_id, bool drop_cache) {
        // Push the HDF5 library cache to the VFD (OS) first so the descriptor
        // sees every buffered byte. This is cheap and never itself durable.
        if (H5Fflush(file_id, H5F_SCOPE_LOCAL) < 0) {
            return false;
        }
#if defined(__unix__) || defined(__APPLE__)
        void* vfd_handle = nullptr;
        if (H5Fget_vfd_handle(file_id, H5P_DEFAULT, &vfd_handle) < 0 ||
            vfd_handle == nullptr) {
            return false;
        }
        const int fd = *static_cast<int*>(vfd_handle);
        if (fd < 0) {
            return false;
        }
        if (fsync(fd) != 0) {
            return false;
        }
        if (drop_cache) {
#if defined(POSIX_FADV_DONTNEED)
            // Best-effort: after fsync the pages are clean, so dropping them
            // keeps periodic writeback from pinning the entire output in RAM.
            // Ignore the return value; failure only forfeits the memory hint.
            (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
        }
        return true;
#else
        (void)drop_cache;
        return false;
#endif
    }

    /// Probe the HDF5 object type at ``group_path`` inside an open file.
    ///
    /// Returns ``H5O_TYPE_GROUP`` for sparse encodings (data/indices/indptr
    /// group layout) and ``H5O_TYPE_DATASET`` for dense 2D arrays.  The caller
    /// is responsible for closing ``file_id``.
    ///
    /// @throws std::runtime_error when the path does not exist.
    inline H5O_type_t probe_object_type(hid_t file_id,
                                        const std::string& group_path,
                                        const char* err_context) {
        H5O_info_t info;
#if H5_VERSION_GE(1, 12, 0)
        herr_t status = H5Oget_info_by_name(file_id, group_path.c_str(), &info,
                                            H5O_INFO_BASIC, H5P_DEFAULT);
#else
        herr_t status = H5Oget_info_by_name(file_id, group_path.c_str(), &info,
                                            H5P_DEFAULT);
#endif
        if (status < 0) {
            throw std::runtime_error(
                std::string(err_context) + ": path not found: " + group_path);
        }
        return info.type;
    }

    // ------------------------------------------------------------------
    // Shared 1-D read helpers.
    //
    // These centralise the hyperslab/point read pattern that the compute
    // operators previously duplicated in anonymous-namespace helpers and in
    // ``read_data_indices_slice_``. All of them use RAII ``Space`` handles so
    // they cannot leak an HDF5 identifier when a read throws.
    // ------------------------------------------------------------------

    /// Read ``count`` contiguous elements starting at ``start`` from a 1-D
    /// dataset into ``destination`` using ``memory_type`` as the in-memory
    /// datatype. ``destination`` must have room for ``count`` elements.
    inline void read_1d_slice_raw(hid_t dataset,
                                  hid_t memory_type,
                                  unsigned long long start,
                                  unsigned long long count,
                                  void* destination,
                                  const char* err_context) {
        if (count == 0) {
            return;
        }
        const hsize_t offset[1] = {static_cast<hsize_t>(start)};
        const hsize_t extent[1] = {static_cast<hsize_t>(count)};
        Space file_space(H5Dget_space(dataset));
        check_h5(static_cast<bool>(file_space),
                 "Failed to open 1-D source dataspace");
        check_h5(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, offset,
                                     nullptr, extent, nullptr) >= 0,
                 "Failed to select 1-D source hyperslab");
        Space mem_space(H5Screate_simple(1, extent, nullptr));
        check_h5(static_cast<bool>(mem_space),
                 "Failed to create 1-D memory dataspace");
        check_h5(H5Dread(dataset, memory_type, mem_space.get(),
                         file_space.get(), H5P_DEFAULT, destination) >= 0,
                 err_context);
    }

    /// Read a contiguous slice of sparse indices into ``std::uint64_t`` values,
    /// dispatching on the stored sign so int32/int64/uint32/uint64 sources are
    /// all handled. Signed sources are checked for negativity; a negative value
    /// throws. This mirrors ``read_1d_indices`` in ``h5ad_matrix_io.cpp``.
    inline std::vector<std::uint64_t> read_indices_slice(
        hid_t dataset,
        unsigned long long start,
        unsigned long long count) {
        std::vector<std::uint64_t> out(static_cast<std::size_t>(count), 0);
        if (count == 0) {
            return out;
        }
        Type type(H5Dget_type(dataset));
        check_h5(static_cast<bool>(type),
                 "Failed to inspect sparse indices dtype");
        if (H5Tget_sign(type.get()) == H5T_SGN_NONE) {
            std::vector<unsigned long long> values(
                static_cast<std::size_t>(count), 0);
            read_1d_slice_raw(dataset, H5T_NATIVE_ULLONG, start, count,
                              values.data(),
                              "Failed to read unsigned sparse indices slice");
            for (std::size_t i = 0; i < values.size(); ++i) {
                out[i] = static_cast<std::uint64_t>(values[i]);
            }
        } else {
            std::vector<long long> values(
                static_cast<std::size_t>(count), 0);
            read_1d_slice_raw(dataset, H5T_NATIVE_LLONG, start, count,
                              values.data(),
                              "Failed to read signed sparse indices slice");
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (values[i] < 0) {
                    throw std::runtime_error(
                        "Sparse indices contain a negative value");
                }
                out[i] = static_cast<std::uint64_t>(values[i]);
            }
        }
        return out;
    }

    /// Read a contiguous slice of sparse ``data`` as ``double``. ``values`` is
    /// resized to ``count``.
    inline void read_double_slice(hid_t dataset,
                                  unsigned long long start,
                                  unsigned long long count,
                                  std::vector<double>& values) {
        values.assign(static_cast<std::size_t>(count), 0.0);
        read_1d_slice_raw(dataset, H5T_NATIVE_DOUBLE, start, count,
                          values.data(), "Failed to read sparse data slice");
    }

    /// Read the individually selected element ``positions`` of a 1-D ``data``
    /// dataset as ``double``. ``values`` is resized to ``positions.size()``.
    inline void read_double_points(hid_t dataset,
                                    const std::vector<hsize_t>& positions,
                                    std::vector<double>& values) {
        values.assign(positions.size(), 0.0);
        if (positions.empty()) {
            return;
        }
        Space file_space(H5Dget_space(dataset));
        check_h5(static_cast<bool>(file_space),
                 "Failed to open sparse data dataspace");
        check_h5(H5Sselect_elements(file_space.get(), H5S_SELECT_SET,
                                    positions.size(), positions.data()) >= 0,
                 "Failed to select sparse data points");
        const hsize_t extent[1] = {static_cast<hsize_t>(positions.size())};
        Space mem_space(H5Screate_simple(1, extent, nullptr));
        check_h5(static_cast<bool>(mem_space),
                 "Failed to create sparse point memory space");
        check_h5(H5Dread(dataset, H5T_NATIVE_DOUBLE, mem_space.get(),
                         file_space.get(), H5P_DEFAULT, values.data()) >= 0,
                 "Failed to read sparse data points");
    }

    /// Validate that a stored, already-nonnegative index fits within ``extent``
    /// and return it as an ``std::uint64_t``. Throws ``std::runtime_error`` with
    /// ``err_context`` when the index is out of bounds. Callers that read via
    /// ``read_indices_slice`` have already rejected negative values.
    inline std::uint64_t validate_and_cast_index(std::uint64_t raw,
                                                 std::uint64_t extent,
                                                 const char* err_context) {
        if (raw >= extent) {
            throw std::runtime_error(err_context);
        }
        return raw;
    }

} // namespace actionet::detail::h5
