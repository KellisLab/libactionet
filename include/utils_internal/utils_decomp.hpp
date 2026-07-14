#ifndef ACTIONET_UTILS_DECOMP_HPP
#define ACTIONET_UTILS_DECOMP_HPP

#include "libactionet_config.hpp"

namespace actionet {

/// @brief Gram-Schmidt orthogonalization.
void gram_schmidt(arma::mat &A);

/// @brief Generate standard normal random matrix.
arma::mat randNorm(int l, int m, int seed);

/// @brief Orient SVD components for sign consistency (modifies in-place).
void orient_SVD(arma::field<arma::mat>& SVD_res);

/// @brief Guard that both per-axis dimensions fit in `INT_MAX`.
///
/// All SVD algorithms in libactionet (IRLB, Halko) narrow the row/col
/// dimensions to `int` for the CBLAS calls on their sketch buffers. Sparse
/// `nnz > INT_MAX` is supported (arma is 64-bit clean under
/// `ARMA_64BIT_WORD`), but per-axis dimensions above `INT_MAX` (~2.1B)
/// would silently overflow the 32-bit BLAS ldm/ldn arguments.
///
/// Throws `std::overflow_error` with a unified message shape:
///   "<label>: matrix dimension exceeds INT_MAX (rows=..., cols=..., INT_MAX=...).
///    Per-axis dimensions above INT_MAX (~2.1B) are not yet supported by any SVD algorithm."
///
/// @param rows  Number of rows.
/// @param cols  Number of columns.
/// @param label Caller identifier (e.g. "svdIRLB (sparse)", "Halko operator").
void check_svd_axis_dimensions(arma::uword rows, arma::uword cols, const char* label);

/// @brief Clamp Halko's target rank `dim` to a legal value given axis dimensions.
///
/// Halko forms a sketch of width `dim + 2` and later trims the trailing two
/// columns, so `dim + 2 <= min(rows, cols)` is required. This helper also
/// enforces `dim >= 1`. Callers must have already invoked
/// `check_svd_axis_dimensions` so that `rows`/`cols` fit in `int`.
///
/// @param rows Number of rows in the input.
/// @param cols Number of columns in the input.
/// @param dim  Requested rank (mutated in place).
void clamp_halko_dim(arma::uword rows, arma::uword cols, int& dim);

} // namespace actionet

#endif //ACTIONET_UTILS_DECOMP_HPP
