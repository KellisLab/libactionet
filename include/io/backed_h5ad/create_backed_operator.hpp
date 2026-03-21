#ifndef ACTIONET_CREATE_BACKED_OPERATOR_HPP
#define ACTIONET_CREATE_BACKED_OPERATOR_HPP

#include "decomposition/matrix_operator.hpp"
#include <memory>
#include <string>
#include <vector>

namespace actionet {

    /// @brief Create the appropriate backed MatrixOperator for an h5ad path.
    ///
    /// Probes the HDF5 object at @p group_path to determine whether it is:
    ///   - a group containing sparse CSR/CSC datasets → BackedSparseMatrixOperator
    ///   - a 2D dataset → BackedDenseMatrixOperator
    ///
    /// @param file_path         Path to the .h5ad file.
    /// @param group_path        HDF5 path (e.g. "/X", "/layers/counts").
    /// @param chunk_size        Upper bound on rows per slab / chunk.
    /// @param row_scale_factors Per-observation scale factors (or empty).
    /// @param apply_log1p       Apply log1p element-wise.
    ///
    /// @return A shared_ptr to the appropriate MatrixOperator subclass.
    std::shared_ptr<MatrixOperator> createBackedOperator(
        const std::string& file_path,
        const std::string& group_path = "/X",
        arma::uword chunk_size = 4096,
        const std::vector<double>& row_scale_factors = {},
        bool apply_log1p = false);

} // namespace actionet

#endif // ACTIONET_CREATE_BACKED_OPERATOR_HPP
