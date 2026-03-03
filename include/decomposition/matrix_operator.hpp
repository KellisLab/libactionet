// Matrix operator abstraction for out-of-memory (OOM) SVD and kernel reduction.
//
// This header defines a lightweight abstract interface for matrix-vector products
// without requiring the full matrix to be materialised in memory.  The primary
// consumer is the PRIMME SVD solver (svd_primme.cpp), which calls matvec/rmatvec
// repeatedly during iterative eigenvalue computation.
//
// Design notes:
//   - Implementations MUST be safe to call from a single thread only.  PRIMME is
//     configured in single-threaded mode for the operator path.  Multi-threaded
//     matvec would require careful coordination with the GIL (Python) or R runtime.
//   - The PythonMatrixOperator subclass (actionet-python) acquires the GIL on
//     each call; see wp_utils.h for details.
//   - Dense and sparse adapters are provided for convenience when the matrix *is*
//     in memory but a uniform operator interface is desired (e.g. testing).

#ifndef ACTIONET_MATRIX_OPERATOR_HPP
#define ACTIONET_MATRIX_OPERATOR_HPP

#include "libactionet_config.hpp"
#include <stdexcept>

namespace actionet {

    /// @brief Abstract matrix operator with forward and transpose products.
    ///
    /// Represents a logical m × n matrix via its action on vectors.
    /// Subclasses must implement matvec (y = A*x) and rmatvec (y = A'*x).
    class MatrixOperator {
    public:
        virtual ~MatrixOperator() = default;

        /// @return Number of rows (m) in the logical matrix.
        virtual arma::uword rows() const = 0;

        /// @return Number of columns (n) in the logical matrix.
        virtual arma::uword cols() const = 0;

        /// @brief Compute y = A * x.
        /// @param x Input vector of length n (cols).
        /// @param y Output vector of length m (rows); will be resized if necessary.
        virtual void matvec(const arma::vec& x, arma::vec& y) const = 0;

        /// @brief Compute y = A' * x  (transpose product).
        /// @param x Input vector of length m (rows).
        /// @param y Output vector of length n (cols); will be resized if necessary.
        virtual void rmatvec(const arma::vec& x, arma::vec& y) const = 0;

        /// @brief Compute Y = A * X for a dense block of vectors.
        ///
        /// Default implementation loops over columns and dispatches to matvec.
        virtual void matmat(const arma::mat& X, arma::mat& Y) const {
            if (X.n_rows != cols()) {
                throw std::runtime_error("MatrixOperator::matmat dimension mismatch");
            }

            Y.set_size(rows(), X.n_cols);
            arma::vec col_out;
            for (arma::uword j = 0; j < X.n_cols; ++j) {
                matvec(X.col(j), col_out);
                Y.col(j) = col_out;
            }
        }

        /// @brief Compute Y = A' * X for a dense block of vectors.
        ///
        /// Default implementation loops over columns and dispatches to rmatvec.
        virtual void rmatmat(const arma::mat& X, arma::mat& Y) const {
            if (X.n_rows != rows()) {
                throw std::runtime_error("MatrixOperator::rmatmat dimension mismatch");
            }

            Y.set_size(cols(), X.n_cols);
            arma::vec col_out;
            for (arma::uword j = 0; j < X.n_cols; ++j) {
                rmatvec(X.col(j), col_out);
                Y.col(j) = col_out;
            }
        }
    };

    /// @brief MatrixOperator adapter for dense Armadillo matrices.
    ///
    /// Holds a non-owning pointer; the underlying matrix must outlive this object.
    class DenseMatrixOperator final : public MatrixOperator {
    public:
        explicit DenseMatrixOperator(const arma::mat& matrix) : matrix_(&matrix) {}

        arma::uword rows() const override { return matrix_->n_rows; }
        arma::uword cols() const override { return matrix_->n_cols; }

        void matvec(const arma::vec& x, arma::vec& y) const override { y = (*matrix_) * x; }
        void rmatvec(const arma::vec& x, arma::vec& y) const override { y = matrix_->t() * x; }
        void matmat(const arma::mat& X, arma::mat& Y) const override { Y = (*matrix_) * X; }
        void rmatmat(const arma::mat& X, arma::mat& Y) const override { Y = matrix_->t() * X; }

    private:
        const arma::mat* matrix_;
    };

    /// @brief MatrixOperator adapter for sparse Armadillo matrices.
    ///
    /// Holds a non-owning pointer; the underlying matrix must outlive this object.
    class SparseMatrixOperator final : public MatrixOperator {
    public:
        explicit SparseMatrixOperator(const arma::sp_mat& matrix) : matrix_(&matrix) {}

        arma::uword rows() const override { return matrix_->n_rows; }
        arma::uword cols() const override { return matrix_->n_cols; }

        void matvec(const arma::vec& x, arma::vec& y) const override { y = (*matrix_) * x; }
        void rmatvec(const arma::vec& x, arma::vec& y) const override { y = matrix_->t() * x; }
        void matmat(const arma::mat& X, arma::mat& Y) const override { Y = (*matrix_) * X; }
        void rmatmat(const arma::mat& X, arma::mat& Y) const override { Y = matrix_->t() * X; }

    private:
        const arma::sp_mat* matrix_;
    };

} // namespace actionet

#endif // ACTIONET_MATRIX_OPERATOR_HPP
