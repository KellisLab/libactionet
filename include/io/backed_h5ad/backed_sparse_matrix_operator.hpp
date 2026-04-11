#ifndef ACTIONET_BACKED_SPARSE_MATRIX_OPERATOR_HPP
#define ACTIONET_BACKED_SPARSE_MATRIX_OPERATOR_HPP

#include "decomposition/matrix_operator.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "fastapprox/fastlog.h"
#include <cmath>
#include <hdf5.h>
#include <memory>
#include <string>
#include <vector>

namespace actionet {

    // Forward declarations for friend functions defined in annotation/specificity.cpp.
    // These non-template overloads implement the backed sparse specificity algorithm
    // and require direct access to BackedSparseMatrixOperator private members.
    class BackedSparseMatrixOperator;
    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::mat& H, int thread_no);
    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     const arma::uvec& labels, int thread_no);
    // Forward declarations for the internal single-pass scan helpers.
    void backed_specificity_scan_csr_(const BackedSparseMatrixOperator& op,
                                      const arma::mat& H_norm_t,
                                      arma::vec& row_count,
                                      arma::vec& col_count,
                                      arma::vec& row_factor_sum_orig,
                                      arma::mat& obs_orig,
                                      double& min_stored,
                                      int thread_no);
    void backed_specificity_scan_csc_(const BackedSparseMatrixOperator& op,
                                      const arma::mat& H_norm_t,
                                      arma::vec& row_count,
                                      arma::vec& col_count,
                                      arma::vec& row_factor_sum_orig,
                                      arma::mat& obs_orig,
                                      double& min_stored,
                                      int thread_no);
    void backed_specificity_support_csr_(const BackedSparseMatrixOperator& op,
                                         const arma::mat& H_norm_t,
                                         arma::mat& obs_out,
                                         double shift,
                                         int thread_no);
    void backed_specificity_support_csc_(const BackedSparseMatrixOperator& op,
                                         const arma::mat& H_norm_t,
                                         arma::mat& obs_out,
                                         double shift,
                                         int thread_no);

    /// @brief MatrixOperator implementation backed by sparse AnnData h5ad storage.
    ///
    /// The underlying h5ad matrix is stored as obs x var (cells x genes).
    /// This operator exposes the native obs x var shape (cells x genes),
    /// matching the AnnData-native orientation contract (Plan 02).
    ///
    /// @par Log-transform approximation
    /// When @c apply_log1p is true, elements are transformed using the
    /// Paul Mineiro @c fastlog() approximation (float precision) instead of
    /// @c std::log1p().  This yields ~0.3% peak relative error over the
    /// typical count-data range and avoids the cost of a libm call per NNZ.
    /// Because the computation is performed in @c float, values above ~16M
    /// lose integer precision after the cast; for standard library-size
    /// normalized single-cell data this is not a concern.
    class BackedSparseMatrixOperator final : public MatrixOperator {
    public:
        BackedSparseMatrixOperator(const std::string& file_path,
                                   const std::string& group_path = "/X",
                                   arma::uword chunk_size = 4096,
                                   const std::vector<double>& row_scale_factors = {},
                                   bool apply_log1p = false,
                                   double log_scale = 1.0,
                                   // Per-read sparse I/O target in bytes. If non-zero, this
                                   // is used directly for NNZ-targeted chunking.
                                   size_t io_target_chunk_bytes = 0,
                                   // Auto-target multiplier used only when
                                   // io_target_chunk_bytes == 0.
                                   //
                                   // Let:
                                   //   bytes_per_nnz = sizeof(double) + sizeof(uint64_t) = 16
                                   //   mean_nnz_axis = total_nnz / axis_len
                                   //   estimated_cap_nnz = chunk_size * mean_nnz_axis
                                   //
                                   // Then:
                                   //   target_chunk_nnz =
                                   //     ceil(io_target_chunk_fraction_of_cap * estimated_cap_nnz)
                                   //   target_chunk_bytes =
                                   //     ceil(target_chunk_nnz * bytes_per_nnz)
                                   //
                                   // The default (0.5) was chosen from coarse atlas-scale
                                   // benchmarks to retain near-cap throughput while reducing
                                   // peak RSS substantially versus full-cap behavior.
                                   double io_target_chunk_fraction_of_cap = 0.5,
                                   int n_threads = 0);
        ~BackedSparseMatrixOperator() override;

        BackedSparseMatrixOperator(const BackedSparseMatrixOperator&) = delete;
        BackedSparseMatrixOperator& operator=(const BackedSparseMatrixOperator&) = delete;
        BackedSparseMatrixOperator(BackedSparseMatrixOperator&& other) noexcept;
        BackedSparseMatrixOperator& operator=(BackedSparseMatrixOperator&& other) noexcept;

        arma::uword rows() const override { return n_obs_; } // rows = cells (obs)
        arma::uword cols() const override { return n_var_; } // cols = genes (var)

        void matvec(const arma::vec& x, arma::vec& y) const override;
        void rmatvec(const arma::vec& x, arma::vec& y) const override;
        void matmat(const arma::mat& X, arma::mat& Y) const override;
        void rmatmat(const arma::mat& X, arma::mat& Y) const override;
        bool prefer_block_solver_for_irlb() const override { return true; }

        const std::string& filePath() const { return file_path_; }
        const std::string& groupPath() const { return group_path_; }
        bool isCSR() const { return is_csr_; }

        /// @brief Extract selected columns as a dense matrix.
        ///
        /// Returns an (n_selected_rows × n_selected_cols) dense matrix containing
        /// the requested columns in request order.  When @p row_indices is non-empty,
        /// only those observation rows are included; otherwise all rows are returned.
        ///
        /// @param col_indices  Column indices to extract (0-based, request-ordered).
        /// @param row_indices  Row indices to extract (0-based); empty = all rows.
        arma::mat takeColumnsDense(const arma::uvec& col_indices,
                                   const arma::uvec& row_indices = {}) const;

        /// @brief Extract selected columns as a sparse matrix.
        ///
        /// Same semantics as takeColumnsDense but returns CSC sparse output.
        arma::sp_mat takeColumnsSparse(const arma::uvec& col_indices,
                                       const arma::uvec& row_indices = {}) const;

        /// @brief Compute fused per-row statistics in a single NNZ pass.
        ///
        /// Streams through the on-disk sparse data once, applying the
        /// lazy transform (row_scale + log1p) before accumulation.
        /// Replaces the former Pass 2 (matvec for row_sum) + Pass 3
        /// (column-chunked dense extraction for row_sum_sq / nnz).
        ///
        /// @param[out] row_sum     Sum of (transformed) values per row.
        /// @param[out] row_sum_sq  Sum of squared (transformed) values per row.
        /// @param[out] nnz         Count of stored (non-zero) entries per row.
        void rowStats(arma::vec& row_sum, arma::vec& row_sum_sq,
                      arma::vec& nnz) const;

    private:
        // Grant direct access to the single-pass backed specificity implementation.
        friend arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                                const arma::mat& H, int thread_no);
        friend arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                                const arma::uvec& labels, int thread_no);
        // Internal scan helpers used by the above friends.
        friend void backed_specificity_scan_csr_(const BackedSparseMatrixOperator& op,
                                                 const arma::mat& H_norm_t,
                                                 arma::vec& row_count,
                                                 arma::vec& col_count,
                                                 arma::vec& row_factor_sum_orig,
                                                 arma::mat& obs_orig,
                                                 double& min_stored,
                                                 int thread_no);
        friend void backed_specificity_scan_csc_(const BackedSparseMatrixOperator& op,
                                                 const arma::mat& H_norm_t,
                                                 arma::vec& row_count,
                                                 arma::vec& col_count,
                                                 arma::vec& row_factor_sum_orig,
                                                 arma::mat& obs_orig,
                                                 double& min_stored,
                                                 int thread_no);
        friend void backed_specificity_support_csr_(const BackedSparseMatrixOperator& op,
                                                    const arma::mat& H_norm_t,
                                                    arma::mat& obs_out,
                                                    double shift,
                                                    int thread_no);
        friend void backed_specificity_support_csc_(const BackedSparseMatrixOperator& op,
                                                    const arma::mat& H_norm_t,
                                                    arma::mat& obs_out,
                                                    double shift,
                                                    int thread_no);
        static std::string read_string_attribute_(hid_t object_id, const char* name);
        static std::vector<long long> read_shape_attribute_(hid_t object_id, const char* name);

        void read_data_indices_slice_(unsigned long long start, unsigned long long count,
                                      std::vector<double>& data, std::vector<unsigned long long>& indices) const;
        void load_chunk_cached_(unsigned long long nnz_start, unsigned long long nnz_count,
                                const std::vector<double>*& data, const std::vector<unsigned long long>*& indices) const;
        void ensure_chunk_transformed_csr_(arma::uword row_start, arma::uword row_end,
                                           unsigned long long nnz_start) const;
        void ensure_chunk_transformed_csc_() const;
        arma::uword next_block_end_(arma::uword start, arma::uword limit) const;
        inline double transform_value_(arma::uword obs_index, double value) const {
            if (no_transform_) return value;
            if (has_row_scale_) value *= row_scale_(obs_index);
            if (apply_log1p_) {
                value = static_cast<double>(fastlog(1.0f + static_cast<float>(value)));
                if (apply_log_scale_) value *= log_scale_;
            }
            return value;
        }
        inline double row_scale_for_(arma::uword obs_index) const {
            return has_row_scale_ ? row_scale_(obs_index) : 1.0;
        }
        inline double transform_scaled_(double value) const {
            if (apply_log1p_) {
                value = static_cast<double>(fastlog(1.0f + static_cast<float>(value)));
                if (apply_log_scale_) value *= log_scale_;
            }
            return value;
        }
        void close_handles_();

        void matvec_csr_(const arma::vec& x, arma::vec& y) const;
        void rmatvec_csr_(const arma::vec& x, arma::vec& y) const;
        void matmat_csr_(const arma::mat& X, arma::mat& Y) const;
        void rmatmat_csr_(const arma::mat& X, arma::mat& Y) const;

        void matvec_csc_(const arma::vec& x, arma::vec& y) const;
        void rmatvec_csc_(const arma::vec& x, arma::vec& y) const;
        void matmat_csc_(const arma::mat& X, arma::mat& Y) const;
        void rmatmat_csc_(const arma::mat& X, arma::mat& Y) const;

        void take_columns_dense_csr_(const arma::uvec& col_indices,
                                     const arma::uvec& row_indices,
                                     arma::mat& out) const;
        void take_columns_dense_csc_(const arma::uvec& col_indices,
                                     const arma::uvec& row_indices,
                                     arma::mat& out) const;

        void row_stats_csr_(arma::vec& row_sum, arma::vec& row_sum_sq,
                            arma::vec& nnz) const;
        void row_stats_csc_(arma::vec& row_sum, arma::vec& row_sum_sq,
                            arma::vec& nnz) const;

        std::string file_path_;
        std::string group_path_;
        bool is_csr_;
        bool apply_log1p_;
        double log_scale_;
        bool apply_log_scale_;
        bool has_row_scale_;
        bool no_transform_;
        arma::uword chunk_size_;
        unsigned long long target_chunk_nnz_;
        arma::uword n_obs_;
        arma::uword n_var_;
        arma::vec row_scale_;
        unsigned int n_threads_;

        hid_t file_id_;
        hid_t group_id_;
        hid_t data_ds_;
        hid_t indices_ds_;
        hid_t indptr_ds_;

        std::vector<unsigned long long> indptr_;

        // LRU-1 chunk cache: avoids re-reading the same contiguous slice on
        // back-to-back matmat/rmatmat calls that share the same chunk window.
        struct ChunkCache {
            unsigned long long start = 0;
            unsigned long long count = 0;
            bool transformed = false;
            std::vector<double> data;
            std::vector<unsigned long long> indices;
        };
        mutable ChunkCache chunk_cache_;
    };
} // namespace actionet

#endif // ACTIONET_BACKED_SPARSE_MATRIX_OPERATOR_HPP
