#ifndef ACTIONET_BACKED_DENSE_MATRIX_OPERATOR_HPP
#define ACTIONET_BACKED_DENSE_MATRIX_OPERATOR_HPP

#include "decomposition/matrix_operator.hpp"
#include <hdf5.h>
#include <memory>
#include <string>
#include <vector>

namespace actionet {

    /// @brief MatrixOperator implementation backed by dense AnnData h5ad storage.
    ///
    /// The underlying h5ad matrix is stored as obs x var (cells x genes).
    /// This operator exposes the native obs x var shape (cells x genes),
    /// matching the AnnData-native orientation contract (Plan 02).
    ///
    /// Dense slabs are read via HDF5 hyperslabs.  The user-supplied chunk_size
    /// is treated as an upper bound on rows-per-slab; the actual slab height is
    /// internally clamped to a byte budget (default 256 MiB) to keep peak memory
    /// bounded regardless of the number of features.
    class BackedDenseMatrixOperator final : public MatrixOperator {
    public:
        /// @param file_path        Path to the .h5ad file.
        /// @param group_path       HDF5 path to the dense dataset (e.g. "/X" or "/layers/counts").
        /// @param chunk_size       Upper bound on observation rows per slab.
        /// @param row_scale_factors Per-observation scale factors (length n_obs, or empty).
        /// @param apply_log1p       Apply log1p transform to each element.
        /// @param slab_byte_budget  Maximum bytes per dense slab buffer (default 256 MiB).
        BackedDenseMatrixOperator(const std::string& file_path,
                                  const std::string& group_path = "/X",
                                  arma::uword chunk_size = 4096,
                                  const std::vector<double>& row_scale_factors = {},
                                  bool apply_log1p = false,
                                  size_t slab_byte_budget = 256ULL * 1024 * 1024);
        ~BackedDenseMatrixOperator() override;

        BackedDenseMatrixOperator(const BackedDenseMatrixOperator&) = delete;
        BackedDenseMatrixOperator& operator=(const BackedDenseMatrixOperator&) = delete;
        BackedDenseMatrixOperator(BackedDenseMatrixOperator&& other) noexcept;
        BackedDenseMatrixOperator& operator=(BackedDenseMatrixOperator&& other) noexcept;

        arma::uword rows() const override { return n_obs_; }  // rows = cells (obs)
        arma::uword cols() const override { return n_var_; }  // cols = genes (var)

        void matvec(const arma::vec& x, arma::vec& y) const override;
        void rmatvec(const arma::vec& x, arma::vec& y) const override;
        void matmat(const arma::mat& X, arma::mat& Y) const override;
        void rmatmat(const arma::mat& X, arma::mat& Y) const override;
        bool prefer_block_solver_for_irlb() const override { return true; }

        const std::string& filePath() const { return file_path_; }
        const std::string& groupPath() const { return group_path_; }
        arma::uword effectiveChunkSize() const { return effective_chunk_size_; }

        /// @brief Extract selected columns as a dense matrix via matmat with a
        ///        selector matrix.
        ///
        /// @param col_indices  Column indices to extract (0-based, request-ordered).
        /// @param row_indices  Row indices to extract (0-based); empty = all rows.
        arma::mat takeColumnsDense(const arma::uvec& col_indices,
                                   const arma::uvec& row_indices = {}) const;

        /// @brief Extract selected columns as a sparse matrix.
        arma::sp_mat takeColumnsSparse(const arma::uvec& col_indices,
                                       const arma::uvec& row_indices = {}) const;

        /// @brief Read an obs-chunk into a dense matrix (obs_count × n_var).
        ///
        /// The slab is read from HDF5 and all configured transforms (log1p,
        /// row_scale) are applied before returning.  The resulting matrix has
        /// @p obs_count rows and n_var columns.
        ///
        /// @param obs_start  First observation row to read.
        /// @param obs_count  Number of rows to read (clamped to available rows).
        /// @param slab       Output matrix; resized to (obs_count × n_var).
        void readSlab(arma::uword obs_start, arma::uword obs_count, arma::mat& slab) const {
            const arma::uword clamped = std::min(obs_count, n_obs_ - obs_start);
            read_slab_(obs_start, clamped, slab);
            apply_transforms_(obs_start, slab);
        }

    private:
        static std::vector<long long> read_shape_(hid_t dataset_id);

        void read_slab_(arma::uword obs_start, arma::uword obs_count,
                        arma::mat& slab) const;
        void apply_transforms_(arma::uword obs_start, arma::mat& slab) const;
        void close_handles_();

        std::string file_path_;
        std::string group_path_;
        bool apply_log1p_;
        arma::uword chunk_size_;
        arma::uword effective_chunk_size_;
        arma::uword n_obs_;
        arma::uword n_var_;
        arma::vec row_scale_;

        hid_t file_id_;
        hid_t dataset_id_;
    };

} // namespace actionet

#endif // ACTIONET_BACKED_DENSE_MATRIX_OPERATOR_HPP
