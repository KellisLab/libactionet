// Singular value decomposition (SVD) algorithms
// IRLB implementation.
#include "decomposition/svd_irbla.hpp"
#include "utils_internal/utils_decomp.hpp"
#include "aarand/aarand.hpp"
#include "blas_deps.hpp"
#include <cstring>
#include <functional>
#include <vector>

namespace actionet {
namespace {

// BLAS-based orthogonalization: Y -= X * (X' * Y), using scratch buffer T of size xn*yn.
void irlb_orthog(double *X, double *Y, double *T, int xm, int xn, int yn) {
    double a = 1, b = 1;
    int inc = 1;
    std::memset(T, 0, xn * yn * sizeof(double));
    cblas_dgemv(CblasColMajor, CblasTrans, xm, xn, a, X, xm, Y, inc, b, T, inc);
    a = -1.0;
    b = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, xm, xn, a, X, xm, T, inc, b, Y, inc);
}

// Convergence test for the IRLB outer iteration.
void irlb_convtests(int Bsz, int n, double tol, double svtol, double Smax,
                    double *svratio, double *residuals, int *k, int *converged, double S) {
    int Len_res = 0;
    for (int j = 0; j < Bsz; j++) {
        if ((std::fabs(residuals[j]) < tol * Smax) && (svratio[j] < svtol))
            Len_res++;
    }

    if (Len_res >= n || S == 0) {
        *converged = 1;
        return;
    }
    if (*k < n + Len_res)
        *k = n + Len_res;

    if (*k > Bsz - 3)
        *k = Bsz - 3;

    if (*k < 1)
        *k = 1;

    *converged = 0;
}

// Fill a buffer with standard normal values.
inline void irlb_StdNorm(double *v, int n, std::mt19937_64& engine) {
    for (int ii = 0; ii < n - 1; ii += 2) {
        auto paired = aarand::standard_normal(engine);
        v[ii] = paired.first;
        v[ii + 1] = paired.second;
    }
    auto paired = aarand::standard_normal(engine);
    v[n - 1] = paired.first;
}

using MatvecFn = std::function<void(char transpose, const double* x, double* out)>;

// Helper: sparse matrix-vector multiplication into raw buffer
void sparse_matvec(char transpose, const arma::sp_mat& A, const double* x, double* out) {
    if (transpose == 'n') {
        arma::vec x_vec(const_cast<double*>(x), A.n_cols, false, true);
        arma::vec out_vec(out, A.n_rows, false, true);
        out_vec = A * x_vec;
    } else {
        arma::vec x_vec(const_cast<double*>(x), A.n_rows, false, true);
        arma::vec out_vec(out, A.n_cols, false, true);
        out_vec = A.t() * x_vec;
    }
}

// Helper: dense matrix-vector multiplication into raw buffer
void dense_matvec(char transpose, const arma::mat& A, const double* x, double* out) {
    if (transpose == 'n') {
        arma::vec x_vec(const_cast<double*>(x), A.n_cols, false, true);
        arma::vec out_vec(out, A.n_rows, false, true);
        out_vec = A * x_vec;
    } else {
        arma::vec x_vec(const_cast<double*>(x), A.n_rows, false, true);
        arma::vec out_vec(out, A.n_cols, false, true);
        out_vec = A.t() * x_vec;
    }
}

// Helper: operator-backed matrix-vector multiplication into raw buffer
void operator_matvec(char transpose, const MatrixOperator& A,
                     const double* x, double* out) {
    if (transpose == 'n') {
        arma::vec x_vec(const_cast<double*>(x), A.cols(), false, true);
        arma::vec out_vec(out, A.rows(), false, true);
        A.matvec(x_vec, out_vec);
    } else {
        arma::vec x_vec(const_cast<double*>(x), A.rows(), false, true);
        arma::vec out_vec(out, A.cols(), false, true);
        A.rmatvec(x_vec, out_vec);
    }
}

// Guard: IRLB narrows the row/col dimensions to `int` for the CBLAS calls on
// the (dim+7)-column sketch buffers. Sparse `nnz` is 64-bit clean via arma
// (see the header comment on `svdIRLB_core`), but a per-axis dimension larger
// than `INT_MAX` would silently overflow the BLAS ldm/ldn arguments.
//
// Delegates to the shared `check_svd_axis_dimensions` helper so IRLB and Halko
// throw the same exception type with the same message shape.

// Unified IRLB core: Lanczos bidiagonalization with implicit restarts.
// The matvec callback abstracts over sparse, dense, and operator-backed matrices.
//
// 64-bit contract:
//   - Row/column dimensions are passed in as `int`. Callers MUST guard their
//     inputs so both dimensions fit in `INT_MAX`; this is enforced by the
//     public overloads below via `check_svd_axis_dimensions`.
//   - Sparse `nnz > INT32_MAX` is fully supported: `arma::sp_mat` uses 64-bit
//     indices under `ARMA_64BIT_WORD` (force-defined for libactionet builds in
//     `libactionet_config.hpp`), and the sparse matvec path routes through
//     `arma::sp_mat::operator*(vec)`, which is 64-bit clean. The CBLAS calls
//     inside this routine operate on the (dim+7)-column sketches
//     (`m * work`, `n * work`), not on the full input, so as long as each axis
//     fits in `INT_MAX / work` (~50M for the default `work ≈ 40`) they stay
//     inside the 32-bit BLAS API.
arma::field<arma::mat> svdIRLB_core(int m, int n, int dim, int iters,
                                    int seed, bool verbose,
                                    const char* label,
                                    const MatvecFn& matvec) {
    dim = std::min(dim, std::min(m, n) - 1);

    if (verbose) {
        stdout_printf("IRLB (%s) -- A: %d x %d\n", label, m, n);
        FLUSH;
    }

    double eps = 3e-13;
    double tol = 1e-05, svtol = 1e-5;

    int work = dim + 7;
    int lwork = 7 * work * (1 + work);

    std::vector<double> s_buf(dim);
    std::vector<double> U_buf(m * work);
    std::vector<double> V_buf(n * work);
    std::vector<double> V1_buf(n * work);
    std::vector<double> U1_buf(m * work);
    std::vector<double> W_buf(m * work);
    std::vector<double> F_buf(n);
    std::vector<double> B_buf(work * work);
    std::vector<double> BU_buf(work * work);
    std::vector<double> BV_buf(work * work);
    std::vector<double> BS_buf(work);
    std::vector<double> res_buf(work);
    std::vector<double> T_buf(lwork);
    std::vector<double> svratio_buf(work);

    double* s = s_buf.data();
    double* U = U_buf.data();
    double* V = V_buf.data();
    double* V1 = V1_buf.data();
    double* U1 = U1_buf.data();
    double* W = W_buf.data();
    double* F = F_buf.data();
    double* B = B_buf.data();
    double* BU = BU_buf.data();
    double* BV = BV_buf.data();
    double* BS = BS_buf.data();
    double* res = res_buf.data();
    double* T = T_buf.data();
    double* svratio = svratio_buf.data();

    arma::mat tmp(B, work, work, false);
    arma::mat BUmat(BU, work, work, false);
    arma::vec BSvec(BS, work, false);
    arma::mat BVmat(BV, work, work, false);

    double d, S, R, R_F, SS;
    double* x;
    int jj, kk;
    int converged;
    int j, k = 0;
    int iter = 0;
    double Smax = 0;

    std::memset(B, 0, work * work * sizeof(double));
    std::memset(svratio, 0, work * sizeof(double));

    double alpha = 1, beta = 0;
    int inc = 1;

    std::mt19937_64 engine(seed);
    irlb_StdNorm(V, n, engine);

    /* Main iteration */
    while (iter < iters) {
        j = 0;

        if (iter == 0) {
            d = cblas_dnrm2(n, V, inc);
            d = 1 / d;
            cblas_dscal(n, d, V, inc);
        }
        else
            j = k;

        x = V + j * n;
        matvec('n', x, W + j * m);

        if (iter > 0)
            irlb_orthog(W, W + j * m, T, m, j, 1);

        S = cblas_dnrm2(m, W + j * m, inc);
        SS = 1.0 / S;
        cblas_dscal(m, SS, W + j * m, inc);

        /* The Lanczos process */
        while (j < work) {
            matvec('t', W + j * m, F);

            SS = -S;
            cblas_daxpy(n, SS, V + j * n, inc, F, inc);
            irlb_orthog(V, F, T, n, j + 1, 1);

            if (j + 1 < work) {
                R_F = cblas_dnrm2(n, F, inc);
                R = 1.0 / R_F;

                if (R_F < eps) {
                    irlb_StdNorm(F, n, engine);

                    irlb_orthog(V, F, T, n, j + 1, 1);
                    R_F = cblas_dnrm2(n, F, inc);
                    R = 1.0 / R_F;
                    R_F = 0;
                }

                std::memmove(V + (j + 1) * n, F, n * sizeof(double));
                cblas_dscal(n, R, V + (j + 1) * n, inc);
                B[j * work + j] = S;
                B[(j + 1) * work + j] = R_F;

                x = V + (j + 1) * n;
                matvec('n', x, W + (j + 1) * m);

                R = -R_F;
                cblas_daxpy(m, R, W + j * m, inc, W + (j + 1) * m, inc);

                irlb_orthog(W, W + (j + 1) * m, T, m, j + 1, 1);
                S = cblas_dnrm2(m, W + (j + 1) * m, inc);
                SS = 1.0 / S;

                if (S < eps) {
                    irlb_StdNorm(W + (j + 1) * m, m, engine);

                    irlb_orthog(W, W + (j + 1) * m, T, m, j + 1, 1);
                    S = cblas_dnrm2(m, W + (j + 1) * m, inc);
                    SS = 1.0 / S;
                    cblas_dscal(m, SS, W + (j + 1) * m, inc);
                    S = 0;
                }
                else
                    cblas_dscal(m, SS, W + (j + 1) * m, inc);
            }
            else {
                B[j * work + j] = S;
            }

            j++;
        }

        arma::svd(BUmat, BSvec, BVmat, tmp, "dc");
        BVmat = arma::trans(BVmat);

        R_F = cblas_dnrm2(n, F, inc);
        R = 1.0 / R_F;
        cblas_dscal(n, R, F, inc);

        if (R_F < eps)
            R_F = 0;

        for (jj = 0; jj < j; ++jj) {
            if (BS[jj] > Smax)
                Smax = BS[jj];
            svratio[jj] = std::fabs(svratio[jj] - BS[jj]) / BS[jj];
        }

        for (kk = 0; kk < j; ++kk)
            res[kk] = R_F * BU[kk * work + (j - 1)];

        irlb_convtests(j, dim, tol, svtol, Smax, svratio, res, &k, &converged, S);
        if (k >= work)
            k = work - 1;
        if (k > dim)
            k = dim;
        if (converged == 1) {
            break;
        }

        for (jj = 0; jj < j; ++jj)
            svratio[jj] = BS[jj];

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, k, j, alpha, V, n,
                    BV, work, beta, V1, n);

        std::memmove(V, V1, n * k * sizeof(double));
        std::memmove(V + n * k, F, n * sizeof(double));

        std::memset(B, 0, work * work * sizeof(double));
        for (jj = 0; jj < k; ++jj) {
            B[jj * work + jj] = BS[jj];
            B[k * work + jj] = res[jj];
        }

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, j, alpha, W, m,
                    BU, work, beta, U1, m);

        std::memmove(W, U1, m * k * sizeof(double));
        iter++;
    }

    /* Results */
    std::memmove(s, BS, dim * sizeof(double));
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, dim, work, alpha, W,
                m, BU, work, beta, U, m);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, dim, work, alpha, V,
                n, BV, work, beta, V1, n);
    std::memmove(V, V1, n * dim * sizeof(double));

    arma::field<arma::mat> out(3);
    out(0) = arma::mat(U, m, dim);
    out(1) = arma::vec(s, dim);
    out(2) = arma::mat(V, n, dim);

    if (converged != 1) {
        stderr_printf("IRLB did NOT converge! Try increasing the number of iterations\n");
        FLUSH;
    }

    actionet::orient_SVD(out);
    return out;
}

} // anonymous namespace

// --- Public overloads: thin wrappers that construct the appropriate matvec ---

arma::field<arma::mat> svdIRLB(const arma::sp_mat& A, int dim, int iters, int seed, bool verbose) {
    check_svd_axis_dimensions(A.n_rows, A.n_cols, "svdIRLB (sparse)");
    MatvecFn mv = [&A](char t, const double* x, double* out) {
        sparse_matvec(t, A, x, out);
    };
    return svdIRLB_core(static_cast<int>(A.n_rows), static_cast<int>(A.n_cols),
                        dim, iters, seed, verbose, "sparse", mv);
}

arma::field<arma::mat> svdIRLB(const arma::mat& A, int dim, int iters, int seed, bool verbose) {
    check_svd_axis_dimensions(A.n_rows, A.n_cols, "svdIRLB (dense)");
    MatvecFn mv = [&A](char t, const double* x, double* out) {
        dense_matvec(t, A, x, out);
    };
    return svdIRLB_core(static_cast<int>(A.n_rows), static_cast<int>(A.n_cols),
                        dim, iters, seed, verbose, "dense", mv);
}

arma::field<arma::mat> svdIRLB(const MatrixOperator& A, int dim,
                                int iters, int seed, bool verbose) {
    check_svd_axis_dimensions(A.rows(), A.cols(), "svdIRLB (operator)");
    MatvecFn mv = [&A](char t, const double* x, double* out) {
        operator_matvec(t, A, x, out);
    };
    return svdIRLB_core(static_cast<int>(A.rows()), static_cast<int>(A.cols()),
                        dim, iters, seed, verbose, "operator", mv);
}

} // namespace actionet
