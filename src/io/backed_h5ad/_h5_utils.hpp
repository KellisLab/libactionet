// SPDX-License-Identifier: GPL-3.0-or-later
//
// Internal HDF5 helpers shared across the backed_h5ad translation units.
// This header is intentionally private (kept under src/, not include/) so it
// does not participate in libactionet's public API.

#pragma once

#include <hdf5.h>
#include <cstddef>
#include <stdexcept>
#include <string>

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

} // namespace actionet::detail::h5
