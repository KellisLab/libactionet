#include "io/backed_h5ad/create_backed_operator.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

#include "_h5_utils.hpp"

#include <hdf5.h>
#include <stdexcept>

namespace actionet {

    std::shared_ptr<MatrixOperator> createBackedOperator(
        const std::string& file_path,
        const std::string& group_path,
        arma::uword chunk_size,
        const std::vector<double>& row_scale_factors,
        bool apply_log1p,
        double log_scale,
        size_t io_target_chunk_bytes,
        double io_target_chunk_fraction_of_cap,
        int n_threads) {

        // Probe the object type once, then close and let the concrete
        // operator reopen the file in its constructor.  Reopening keeps the
        // operator RAII-owned and lets it hold the exact HDF5 handles it
        // needs (sparse: group + 3 datasets; dense: dataset only).
        hid_t file_id = detail::h5::open_h5_readonly_no_lock(
            file_path, "createBackedOperator");
        H5O_type_t obj_type;
        try {
            obj_type = detail::h5::probe_object_type(
                file_id, group_path, "createBackedOperator");
        } catch (...) {
            H5Fclose(file_id);
            throw;
        }
        H5Fclose(file_id);

        if (obj_type == H5O_TYPE_GROUP) {
            return std::make_shared<BackedSparseMatrixOperator>(
                file_path,
                group_path,
                chunk_size,
                row_scale_factors,
                apply_log1p,
                log_scale,
                io_target_chunk_bytes,
                io_target_chunk_fraction_of_cap,
                n_threads);
        } else if (obj_type == H5O_TYPE_DATASET) {
            const size_t slab_budget = io_target_chunk_bytes > 0
                ? io_target_chunk_bytes
                : 256ULL * 1024 * 1024;
            return std::make_shared<BackedDenseMatrixOperator>(
                file_path,
                group_path,
                chunk_size,
                row_scale_factors,
                apply_log1p,
                log_scale,
                slab_budget,
                n_threads);
        }
        throw std::runtime_error(
            "createBackedOperator: HDF5 object at '" + group_path +
            "' is neither a group (sparse) nor a dataset (dense)");
    }

} // namespace actionet
