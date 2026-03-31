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
    /// @param log_scale         Scalar multiplier applied after log1p transform.
    ///                          Defaults to 1.0 (natural log behavior).
    /// @param io_target_chunk_bytes Approximate target bytes per sparse read chunk.
    ///                              Set >0 to force an explicit byte target.
    ///                              Set 0 to enable automatic NNZ-targeted chunking.
    /// @param io_target_chunk_fraction_of_cap Auto-target multiplier used when
    ///                              io_target_chunk_bytes == 0.
    /// @param n_threads         OpenMP thread count hint for backed operator compute
    ///                          loops (0 = auto, 1 = serial).
    ///
    /// Auto-target math for sparse inputs:
    ///   bytes_per_nnz = sizeof(double) + sizeof(uint64_t) = 16
    ///   mean_nnz_axis = total_nnz / axis_len
    ///   estimated_cap_nnz = chunk_size * mean_nnz_axis
    ///   target_chunk_nnz = ceil(io_target_chunk_fraction_of_cap * estimated_cap_nnz)
    ///   target_chunk_bytes = ceil(target_chunk_nnz * bytes_per_nnz)
    ///
    /// Default fraction is 0.5 (picked empirically for a memory-first default
    /// that preserves near-cap throughput on atlas-scale backed SVD workloads).
    ///
    /// @return A shared_ptr to the appropriate MatrixOperator subclass.
    std::shared_ptr<MatrixOperator> createBackedOperator(
        const std::string& file_path,
        const std::string& group_path = "/X",
        arma::uword chunk_size = 4096,
        const std::vector<double>& row_scale_factors = {},
        bool apply_log1p = false,
        double log_scale = 1.0,
        size_t io_target_chunk_bytes = 0,
        double io_target_chunk_fraction_of_cap = 0.5,
        int n_threads = 0);

} // namespace actionet

#endif // ACTIONET_CREATE_BACKED_OPERATOR_HPP
