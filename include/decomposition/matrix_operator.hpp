// Matrix operator abstraction for out-of-memory (OOM) SVD and kernel reduction.
//
// This header defines a lightweight abstract interface for matrix-vector products
// without requiring the full matrix to be materialised in memory. The primary
// consumers are the operator overloads of the randomized/Lanczos SVD algorithms
// (svd_halko.cpp, svd_feng.cpp, svd_irbla.cpp), which call matvec/rmatvec and
// matmat/rmatmat repeatedly during iterative solves.
//
// Design notes:
//   - Implementations MUST be safe to call from a single thread only. Multi-
//     threaded matvec would require careful coordination with the GIL (Python)
//     or R runtime.
//   - pybind11 consumers (actionet-python) pass concrete backed subclasses
//     (BackedSparseMatrixOperator, BackedDenseMatrixOperator) as
//     std::shared_ptr<MatrixOperator> and downcast via dispatch_backed_op()
//     in wp_utils.h; no Python-side subclass of this interface exists.
//   - Dense and sparse adapters are provided for convenience when the matrix *is*
//     in memory but a uniform operator interface is desired (e.g. testing).
//   - matmat / rmatmat are pure-virtual: every subclass must provide an
//     efficient blocked implementation. The base class does not fall back to
//     looping matvec because that fallback silently penalised subclasses that
//     forgot to override (particularly relevant for a future GPU operator
//     that must choose between batched GEMM and gemv looping explicitly).

#ifndef ACTIONET_MATRIX_OPERATOR_HPP
#define ACTIONET_MATRIX_OPERATOR_HPP

#include "libactionet_config.hpp"

namespace actionet {

    /// @brief Abstract matrix operator with forward and transpose products.
    ///
    /// Represents a logical m × n matrix via its action on vectors.
    /// Subclasses must implement matvec (y = A*x), rmatvec (y = A'*x),
    /// matmat (Y = A*X), and rmatmat (Y = A'*X). The block methods have
    /// no default fallback; see the per-method docs for the rationale.
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
        /// Implementations MUST provide an efficient blocked kernel. The base
        /// class does not fall back to looping over matvec: any perf-sensitive
        /// subclass that forgot to override would silently pay O(k) matvec
        /// overhead. Future GPU operators must decide whether to dispatch a
        /// batched GEMM or loop over cuBLAS gemv here explicitly.
        virtual void matmat(const arma::mat& X, arma::mat& Y) const = 0;

        /// @brief Compute Y = A' * X for a dense block of vectors.
        ///
        /// Same contract as matmat: implementations MUST provide an efficient
        /// blocked kernel.
        virtual void rmatmat(const arma::mat& X, arma::mat& Y) const = 0;
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
