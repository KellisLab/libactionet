#include "io/backed_h5ad/create_backed_operator.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

#include <hdf5.h>
#include <stdexcept>

namespace actionet {

    std::shared_ptr<MatrixOperator> createBackedOperator(
        const std::string& file_path,
        const std::string& group_path,
        arma::uword chunk_size,
        const std::vector<double>& row_scale_factors,
        bool apply_log1p) {

        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        if (fapl < 0) {
            throw std::runtime_error("createBackedOperator: failed to create file access plist");
        }
        H5Pset_file_locking(fapl, 0, 1);
        hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        if (file_id < 0) {
            throw std::runtime_error("createBackedOperator: failed to open h5ad file: " + file_path);
        }

        H5O_info_t info;
#if H5_VERSION_GE(1, 12, 0)
        herr_t status = H5Oget_info_by_name(file_id, group_path.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT);
#else
        herr_t status = H5Oget_info_by_name(file_id, group_path.c_str(), &info, H5P_DEFAULT);
#endif
        if (status < 0) {
            H5Fclose(file_id);
            throw std::runtime_error("createBackedOperator: path not found: " + group_path);
        }

        H5O_type_t obj_type = info.type;
        H5Fclose(file_id);

        if (obj_type == H5O_TYPE_GROUP) {
            return std::make_shared<BackedSparseMatrixOperator>(
                file_path, group_path, chunk_size, row_scale_factors, apply_log1p);
        } else if (obj_type == H5O_TYPE_DATASET) {
            return std::make_shared<BackedDenseMatrixOperator>(
                file_path, group_path, chunk_size, row_scale_factors, apply_log1p);
        } else {
            throw std::runtime_error(
                "createBackedOperator: HDF5 object at '" + group_path +
                "' is neither a group (sparse) nor a dataset (dense)");
        }
    }

} // namespace actionet
