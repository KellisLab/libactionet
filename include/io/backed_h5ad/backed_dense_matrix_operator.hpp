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
    /// The underlying h5ad matrix is stored as obs x var (a 2D HDF5 dataset).
    /// This operator exposes the transpose shape (var x obs) to match ACTIONet's
    /// decomposition expectations, consistent with BackedSparseMatrixOperator.
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

        arma::uword rows() const override { return n_var_; }
        arma::uword cols() const override { return n_obs_; }

        void matvec(const arma::vec& x, arma::vec& y) const override;
        void rmatvec(const arma::vec& x, arma::vec& y) const override;
        void matmat(const arma::mat& X, arma::mat& Y) const override;
        void rmatmat(const arma::mat& X, arma::mat& Y) const override;

        const std::string& filePath() const { return file_path_; }
        const std::string& groupPath() const { return group_path_; }
        arma::uword effectiveChunkSize() const { return effective_chunk_size_; }

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
