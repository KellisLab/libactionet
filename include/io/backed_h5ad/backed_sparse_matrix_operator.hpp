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
                                                     arma::mat& H, int thread_no);
    arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                     arma::uvec& labels, int thread_no);

    /// @brief MatrixOperator implementation backed by sparse AnnData h5ad storage.
    ///
    /// The underlying h5ad matrix is stored as obs x var. This operator exposes the
    /// transpose shape (var x obs) to match ACTIONet's decomposition expectations.
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

        arma::uword rows() const override { return n_var_; } // S = X' => rows are features
        arma::uword cols() const override { return n_obs_; } // columns are cells

        void matvec(const arma::vec& x, arma::vec& y) const override;
        void rmatvec(const arma::vec& x, arma::vec& y) const override;
        void matmat(const arma::mat& X, arma::mat& Y) const override;
        void rmatmat(const arma::mat& X, arma::mat& Y) const override;

        const std::string& filePath() const { return file_path_; }
        const std::string& groupPath() const { return group_path_; }
        bool isCSR() const { return is_csr_; }

    private:
        // Grant direct access to the single-pass backed specificity implementation.
        friend arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                                arma::mat& H, int thread_no);
        friend arma::field<arma::mat> computeFeatureSpecificity(BackedSparseMatrixOperator& op,
                                                                arma::uvec& labels, int thread_no);
        static std::string read_string_attribute_(hid_t object_id, const char* name);
        static std::vector<long long> read_shape_attribute_(hid_t object_id, const char* name);

        void read_data_indices_slice_(unsigned long long start, unsigned long long count,
                                      std::vector<double>& data, std::vector<unsigned long long>& indices) const;
        void load_chunk_cached_(unsigned long long nnz_start, unsigned long long nnz_count,
                                std::vector<double>& data, std::vector<unsigned long long>& indices) const;
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
