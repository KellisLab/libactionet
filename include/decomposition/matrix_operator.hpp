// Matrix operator abstraction for out-of-memory SVD and kernel reduction
#ifndef ACTIONET_MATRIX_OPERATOR_HPP
#define ACTIONET_MATRIX_OPERATOR_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Abstract matrix operator with forward and transpose products.
    class MatrixOperator {
    public:
        virtual ~MatrixOperator() = default;

        /// @return Number of rows in the logical matrix.
        virtual arma::uword rows() const = 0;

        /// @return Number of columns in the logical matrix.
        virtual arma::uword cols() const = 0;

        /// @brief Compute y = A * x.
        virtual void matvec(const arma::vec& x, arma::vec& y) const = 0;

        /// @brief Compute y = A' * x.
        virtual void rmatvec(const arma::vec& x, arma::vec& y) const = 0;
    };

    /// @brief MatrixOperator adapter for dense Armadillo matrices.
    class DenseMatrixOperator final : public MatrixOperator {
    public:
        explicit DenseMatrixOperator(const arma::mat& matrix) : matrix_(&matrix) {}

        arma::uword rows() const override { return matrix_->n_rows; }
        arma::uword cols() const override { return matrix_->n_cols; }

        void matvec(const arma::vec& x, arma::vec& y) const override { y = (*matrix_) * x; }
        void rmatvec(const arma::vec& x, arma::vec& y) const override { y = matrix_->t() * x; }

    private:
        const arma::mat* matrix_;
    };

    /// @brief MatrixOperator adapter for sparse Armadillo matrices.
    class SparseMatrixOperator final : public MatrixOperator {
    public:
        explicit SparseMatrixOperator(const arma::sp_mat& matrix) : matrix_(&matrix) {}

        arma::uword rows() const override { return matrix_->n_rows; }
        arma::uword cols() const override { return matrix_->n_cols; }

        void matvec(const arma::vec& x, arma::vec& y) const override { y = (*matrix_) * x; }
        void rmatvec(const arma::vec& x, arma::vec& y) const override { y = matrix_->t() * x; }

    private:
        const arma::sp_mat* matrix_;
    };
} // namespace actionet

#endif // ACTIONET_MATRIX_OPERATOR_HPP
