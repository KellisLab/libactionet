#ifndef ACTIONET_BACKED_DENSE_MATRIX_OPERATOR_HPP
#define ACTIONET_BACKED_DENSE_MATRIX_OPERATOR_HPP

#include "decomposition/matrix_operator.hpp"
#include "utils_internal/utils_parallel.hpp"
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
    ///
    /// @par Log-transform approximation
    /// When @c apply_log1p is true, elements are transformed using the
    /// Paul Mineiro @c fastlog() approximation (float precision) instead of
    /// @c std::log1p().  This yields ~0.3% peak relative error over the
    /// typical count-data range and avoids the cost of a libm call per NNZ.
    /// Because the computation is performed in @c float, values above ~16M
    /// lose integer precision after the cast; for standard library-size
    /// normalized single-cell data this is not a concern.
    ///
    /// @par Lazy transform ordering
    /// Every value read from disk is transformed by @c apply_transforms_ as
    /// @code v_out = (apply_log1p ? log1p(row_scale * v_in) : row_scale * v_in) * log_scale @endcode
    /// with the row_scale factor determined by the value's originating obs
    /// row.  Because the underlying h5ad data is dense, on-zero entries are
    /// affected too: with @c apply_log1p the returned value on a stored zero
    /// is @c fastlog(1+0) ~ -1.65e-6 (not exact 0), which is well within the
    /// approximation's stated ~0.3% relative error budget.
    ///
    /// @par Thread-safety
    /// All @c const methods (@c matvec, @c rmatvec, @c matmat, @c rmatmat,
    /// @c takeColumnsDense, @c takeColumnsSparse) are safe to call on
    /// different operator instances concurrently.  A @b single instance is
    /// @b not re-entrant across threads because the internal slab cache is
    /// mutable.  OpenMP is used internally to parallelise the log1p pass and
    /// scatter/gather loops after a slab has been loaded; this is safe
    /// because worker threads only read the cached buffer.  Callers must not
    /// invoke a public method on the operator from multiple host threads
    /// simultaneously.
    class BackedDenseMatrixOperator final : public MatrixOperator {
    public:
        /// @param file_path        Path to the .h5ad file.
        /// @param group_path       HDF5 path to the dense dataset (e.g. "/X" or "/layers/counts").
        /// @param chunk_size       Upper bound on observation rows per slab.
        /// @param row_scale_factors Per-observation scale factors (length n_obs, or empty).
        /// @param apply_log1p       Apply log1p transform to each element.
        /// @param log_scale         Scalar multiplier applied after log1p.
        ///                          Defaults to 1.0.
        /// @param slab_byte_budget  Maximum bytes per dense slab buffer (default 256 MiB).
        /// @param n_threads         OpenMP thread count hint for compute loops (0 = auto, 1 = serial).
        BackedDenseMatrixOperator(const std::string& file_path,
                                  const std::string& group_path = "/X",
                                  arma::uword chunk_size = 4096,
                                  const std::vector<double>& row_scale_factors = {},
                                  bool apply_log1p = false,
                                  double log_scale = 1.0,
                                  size_t slab_byte_budget = 256ULL * 1024 * 1024,
                                  int n_threads = 0);
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

        /// @brief Compute fused per-row statistics in a single slab pass.
        ///
        /// Reads each observation-chunk once (applying the configured
        /// row_scale + log1p transform), accumulating row_sum, row_sum_sq
        /// and the count of nonzero entries. Signature mirrors the
        /// BackedSparseMatrixOperator counterpart so callers can be
        /// templated across the two operator types.
        ///
        /// @param[out] row_sum     Sum of (transformed) values per row.
        /// @param[out] row_sum_sq  Sum of squared (transformed) values per row.
        /// @param[out] nnz         Count of nonzero entries per row.
        void rowStats(arma::vec& row_sum, arma::vec& row_sum_sq,
                      arma::vec& nnz) const;

    private:
        static std::vector<long long> read_shape_(hid_t dataset_id);

        void read_slab_(arma::uword obs_start, arma::uword obs_count,
                        arma::mat& slab) const;
        void apply_transforms_(arma::uword obs_start, arma::mat& slab) const;
        void close_handles_();

        std::string file_path_;
        std::string group_path_;
        bool apply_log1p_;
        double log_scale_;
        arma::uword chunk_size_;
        arma::uword effective_chunk_size_;
        arma::uword n_obs_;
        arma::uword n_var_;
        arma::vec row_scale_;
        unsigned int n_threads_;

        hid_t file_id_;
        hid_t dataset_id_;
    };

} // namespace actionet

#endif // ACTIONET_BACKED_DENSE_MATRIX_OPERATOR_HPP
