#ifndef ACTIONET_UTILS_SMALL_DENSE_HPP
#define ACTIONET_UTILS_SMALL_DENSE_HPP

#include "blas_deps.hpp"
#include "libactionet_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace actionet::utils_internal::small_dense {

// BLAS dispatch overhead and threaded-BLAS coordination dominate ACTION's
// high-frequency skinny products.  Keep those products inside the outer
// OpenMP decomposition; general dense matrices still use the configured BLAS.
inline constexpr arma::uword INLINE_DIMENSION_LIMIT = 128;

inline bool use_inline_kernel(arma::uword rows, arma::uword cols) noexcept {
    return std::min(rows, cols) <= INLINE_DIMENSION_LIMIT;
}

inline void copy(int n, const double* x, int incx, double* y, int incy,
                 bool inline_kernel) {
    if (!inline_kernel) {
        cblas_dcopy(n, x, incx, y, incy);
        return;
    }
    for (int i = 0; i < n; ++i)
        y[i * incy] = x[i * incx];
}

inline double dot(int n, const double* x, int incx, const double* y, int incy,
                  bool inline_kernel) {
    if (!inline_kernel)
        return cblas_ddot(n, x, incx, y, incy);

    double result = 0.0;
    #pragma omp simd reduction(+:result)
    for (int i = 0; i < n; ++i)
        result += x[i * incx] * y[i * incy];
    return result;
}

inline void scale(int n, double alpha, double* x, int incx, bool inline_kernel) {
    if (!inline_kernel) {
        cblas_dscal(n, alpha, x, incx);
        return;
    }
    #pragma omp simd
    for (int i = 0; i < n; ++i)
        x[i * incx] *= alpha;
}

inline void axpy(int n, double alpha, const double* x, int incx, double* y,
                 int incy, bool inline_kernel) {
    if (!inline_kernel) {
        cblas_daxpy(n, alpha, x, incx, y, incy);
        return;
    }
    #pragma omp simd
    for (int i = 0; i < n; ++i)
        y[i * incy] += alpha * x[i * incx];
}

inline void prepare_output(int n, double beta, double* y) {
    if (beta == 0.0) {
        std::fill_n(y, n, 0.0);
    } else if (beta != 1.0) {
        scale(n, beta, y, 1, true);
    }
}

inline void gemv(bool transpose, int rows, int cols, double alpha,
                 const double* matrix, int leading_dimension,
                 const double* x, double beta, double* y,
                 bool inline_kernel) {
    if (!inline_kernel) {
        cblas_dgemv(CblasColMajor,
                    transpose ? CblasTrans : CblasNoTrans,
                    rows, cols, alpha, matrix, leading_dimension,
                    x, 1, beta, y, 1);
        return;
    }

    if (transpose) {
        for (int col = 0; col < cols; ++col) {
            const double* matrix_col = matrix + col * leading_dimension;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int row = 0; row < rows; ++row)
                sum += matrix_col[row] * x[row];
            y[col] = alpha * sum + beta * y[col];
        }
        return;
    }

    prepare_output(rows, beta, y);
    for (int col = 0; col < cols; ++col)
        axpy(rows, alpha * x[col], matrix + col * leading_dimension, 1,
             y, 1, true);
}

inline void symv_upper(int n, double alpha, const double* matrix,
                       int leading_dimension, const double* x, double beta,
                       double* y, bool inline_kernel) {
    if (!inline_kernel) {
        cblas_dsymv(CblasColMajor, CblasUpper, n, alpha, matrix,
                    leading_dimension, x, 1, beta, y, 1);
        return;
    }

    prepare_output(n, beta, y);
    for (int col = 0; col < n; ++col) {
        const double* matrix_col = matrix + col * leading_dimension;
        const double scaled_x = alpha * x[col];
        double off_diagonal = 0.0;
        for (int row = 0; row < col; ++row) {
            y[row] += scaled_x * matrix_col[row];
            off_diagonal += matrix_col[row] * x[row];
        }
        y[col] += scaled_x * matrix_col[col] + alpha * off_diagonal;
    }
}

inline void rank_one_update(int rows, int cols, double alpha,
                            const double* x, const double* y, double* matrix,
                            int leading_dimension, bool inline_kernel) {
    if (!inline_kernel) {
        cblas_dger(CblasColMajor, rows, cols, alpha, x, 1, y, 1,
                   matrix, leading_dimension);
        return;
    }

    for (int col = 0; col < cols; ++col)
        axpy(rows, alpha * y[col], x, 1, matrix + col * leading_dimension,
             1, true);
}

inline arma::mat gram(const arma::mat& matrix, double element_offset = 0.0) {
    if (!use_inline_kernel(matrix.n_rows, matrix.n_cols))
        return arma::trans(matrix) * matrix + element_offset;

    arma::mat result(matrix.n_cols, matrix.n_cols);
    const int rows = static_cast<int>(matrix.n_rows);
    for (arma::uword col = 0; col < matrix.n_cols; ++col) {
        const double* right = matrix.colptr(col);
        for (arma::uword row = 0; row <= col; ++row) {
            const double value = dot(rows, matrix.colptr(row), 1, right, 1, true)
                               + element_offset;
            result(row, col) = value;
            result(col, row) = value;
        }
    }
    return result;
}

inline arma::mat residual_product(const arma::mat& data, const arma::mat& left,
                                  const arma::mat& right) {
    if (!use_inline_kernel(left.n_rows, left.n_cols) &&
        !use_inline_kernel(right.n_rows, right.n_cols))
        return data - left * right;

    arma::mat residual = data;
    const int rows = static_cast<int>(data.n_rows);
    for (arma::uword col = 0; col < data.n_cols; ++col) {
        double* output = residual.colptr(col);
        for (arma::uword inner = 0; inner < left.n_cols; ++inner)
            axpy(rows, -right(inner, col), left.colptr(inner), 1,
                 output, 1, true);
    }
    return residual;
}

inline double frobenius_norm(const arma::mat& matrix) {
    if (!use_inline_kernel(matrix.n_rows, matrix.n_cols))
        return arma::norm(matrix, "fro");

    double squared_norm = 0.0;
    const double* values = matrix.memptr();
    #pragma omp simd reduction(+:squared_norm)
    for (arma::uword index = 0; index < matrix.n_elem; ++index)
        squared_norm += values[index] * values[index];
    return std::sqrt(squared_norm);
}

inline void clamp_and_normalize_columns(arma::mat& matrix, double lower = 0.0,
                                        double upper = 1.0) {
    if (!use_inline_kernel(matrix.n_rows, matrix.n_cols)) {
        matrix = arma::clamp(matrix, lower, upper);
        matrix = arma::normalise(matrix, 1);
        return;
    }

    for (arma::uword col = 0; col < matrix.n_cols; ++col) {
        double* values = matrix.colptr(col);
        double column_norm = 0.0;
        for (arma::uword row = 0; row < matrix.n_rows; ++row) {
            values[row] = std::clamp(values[row], lower, upper);
            column_norm += std::abs(values[row]);
        }
        if (column_norm > 0.0)
            scale(static_cast<int>(matrix.n_rows), 1.0 / column_norm,
                  values, 1, true);
    }
}

} // namespace actionet::utils_internal::small_dense

#endif // ACTIONET_UTILS_SMALL_DENSE_HPP
