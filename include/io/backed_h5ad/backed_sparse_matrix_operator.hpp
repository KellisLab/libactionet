#ifndef ACTIONET_BACKED_SPARSE_MATRIX_OPERATOR_HPP
#define ACTIONET_BACKED_SPARSE_MATRIX_OPERATOR_HPP

#include "decomposition/matrix_operator.hpp"
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
                                      arma::mat& support_obs,
                                      double& min_stored);
    void backed_specificity_scan_csc_(const BackedSparseMatrixOperator& op,
                                      const arma::mat& H_norm_t,
                                      arma::vec& row_count,
                                      arma::vec& col_count,
                                      arma::vec& row_factor_sum_orig,
                                      arma::mat& obs_orig,
                                      arma::mat& support_obs,
                                      double& min_stored);

    /// @brief MatrixOperator implementation backed by sparse AnnData h5ad storage.
    ///
    /// The underlying h5ad matrix is stored as obs x var (cells x genes).
    /// This operator exposes the native obs x var shape (cells x genes),
    /// matching the AnnData-native orientation contract (Plan 02).
    class BackedSparseMatrixOperator final : public MatrixOperator {
    public:
        BackedSparseMatrixOperator(const std::string& file_path,
                                   const std::string& group_path = "/X",
                                   arma::uword chunk_size = 4096,
                                   const std::vector<double>& row_scale_factors = {},
                                   bool apply_log1p = false);
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
                                                 arma::mat& support_obs,
                                                 double& min_stored);
        friend void backed_specificity_scan_csc_(const BackedSparseMatrixOperator& op,
                                                 const arma::mat& H_norm_t,
                                                 arma::vec& row_count,
                                                 arma::vec& col_count,
                                                 arma::vec& row_factor_sum_orig,
                                                 arma::mat& obs_orig,
                                                 arma::mat& support_obs,
                                                 double& min_stored);
        static std::string read_string_attribute_(hid_t object_id, const char* name);
        static std::vector<long long> read_shape_attribute_(hid_t object_id, const char* name);

        void read_data_indices_slice_(unsigned long long start, unsigned long long count,
                                      std::vector<double>& data, std::vector<unsigned long long>& indices) const;
        void load_chunk_cached_(unsigned long long nnz_start, unsigned long long nnz_count,
                                const std::vector<double>*& data, const std::vector<unsigned long long>*& indices) const;
        double transform_value_(arma::uword obs_index, double value) const;
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

        std::string file_path_;
        std::string group_path_;
        bool is_csr_;
        bool apply_log1p_;
        arma::uword chunk_size_;
        arma::uword n_obs_;
        arma::uword n_var_;
        arma::vec row_scale_;

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
            std::vector<double> data;
            std::vector<unsigned long long> indices;
        };
        mutable ChunkCache chunk_cache_;
    };
} // namespace actionet

#endif // ACTIONET_BACKED_SPARSE_MATRIX_OPERATOR_HPP
