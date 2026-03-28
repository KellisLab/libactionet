// Singular value decomposition (SVD) using PRIMME_SVDS
//
// Provides three matvec flavours (sparse, dense, operator) unified through a
// common PRIMME core routine.  The operator path is the primary entry point for
// out-of-memory (OOM) SVD used by PythonMatrixOperator.

#include "decomposition/svd_primme.hpp"
#include "decomposition/svd_main.hpp"
#include "utils_internal/utils_decomp.hpp"
#include "blas_deps.hpp"
#include "primme.h"
#include "primme_svds.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

    // ---- Context wrapper for operator-backed matvec --------------------------------------
    struct PrimmeOperatorCtx {
        const actionet::MatrixOperator* op;
    };

    // ---- Unified PRIMME matvec callback --------------------------------------------------
    //
    // All three matrix flavours (sparse, dense, operator) share the same loop structure.
    // The actual dispatch is performed by a std::function-like approach using a thin
    // callback pair (forward / transpose) extracted at the call site.

    using MatvecDispatch = void (*)(void* ctx, const arma::vec& x, arma::vec& y, bool transpose);

    static void sparseDispatch(void* ctx, const arma::vec& x, arma::vec& y, bool transpose) {
        auto* A = static_cast<arma::sp_mat*>(ctx);
        if (!transpose) {
            y = (*A) * x;
        } else {
            y = A->t() * x;
        }
    }

    static void denseDispatch(void* ctx, const arma::vec& x, arma::vec& y, bool transpose) {
        auto* A = static_cast<arma::mat*>(ctx);
        if (!transpose) {
            y = (*A) * x;
        } else {
            y = A->t() * x;
        }
    }

    static void operatorDispatch(void* ctx, const arma::vec& x, arma::vec& y, bool transpose) {
        auto* op_ctx = static_cast<PrimmeOperatorCtx*>(ctx);
        const actionet::MatrixOperator& op = *(op_ctx->op);
        if (!transpose) {
            op.matvec(x, y);
        } else {
            op.rmatvec(x, y);
        }
    }

    // ---- Generic PRIMME callback ---------------------------------------------------------
    //
    // Stored as a static function pointer pair via the matrix pointer field.  The actual
    // context pointer and dispatch function are packed into a small struct below.

    struct PrimmeCallbackCtx {
        void* user_ctx;
        MatvecDispatch dispatch;
        arma::uword m;
        arma::uword n;
        const actionet::MatrixOperator* op_block;
    };

    static void primmeMatvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy, int* blockSize,
                             int* trans, primme_svds_params* primme_svds, int* err) {
        auto* ctx = static_cast<PrimmeCallbackCtx*>(primme_svds->matrix);

        double* xvec = static_cast<double*>(x);
        double* yvec = static_cast<double*>(y);

        const arma::uword x_len = (*trans == 0) ? ctx->n : ctx->m;
        const arma::uword y_len = (*trans == 0) ? ctx->m : ctx->n;

        if (*blockSize > 1 && ctx->op_block != nullptr) {
            const bool packed = (*ldx == static_cast<PRIMME_INT>(x_len) &&
                                 *ldy == static_cast<PRIMME_INT>(y_len));

            if (packed) {
                arma::mat X(xvec, x_len, static_cast<arma::uword>(*blockSize), false, true);
                arma::mat Y(yvec, y_len, static_cast<arma::uword>(*blockSize), false, true);
                if (*trans == 0) {
                    ctx->op_block->matmat(X, Y);
                }
                else {
                    ctx->op_block->rmatmat(X, Y);
                }
                *err = 0;
                return;
            }

            arma::mat X(x_len, static_cast<arma::uword>(*blockSize));
            for (int i = 0; i < *blockSize; ++i) {
                std::memcpy(X.colptr(static_cast<arma::uword>(i)),
                            xvec + (*ldx) * i,
                            x_len * sizeof(double));
            }

            arma::mat Y;
            if (*trans == 0) {
                ctx->op_block->matmat(X, Y);
            }
            else {
                ctx->op_block->rmatmat(X, Y);
            }

            for (int i = 0; i < *blockSize; ++i) {
                std::memcpy(yvec + (*ldy) * i,
                            Y.colptr(static_cast<arma::uword>(i)),
                            y_len * sizeof(double));
            }
            *err = 0;
            return;
        }

        for (int i = 0; i < *blockSize; i++) {
            arma::vec x_col(xvec + (*ldx) * i, x_len, false, true);
            arma::vec y_col(yvec + (*ldy) * i, y_len, false, true);

            ctx->dispatch(ctx->user_ctx, x_col, y_col, *trans != 0);
        }

        *err = 0;
    }

    // ---- Shared PRIMME core routine ------------------------------------------------------

    actionet::SVDResult runPrimmeCore(PRIMME_INT m, PRIMME_INT n, int k, int max_it, int seed, bool verbose,
                                      PrimmeCallbackCtx* cb_ctx,
                                      const char* label, unsigned long long nnz = 0) {
        actionet::SVDResult empty_out;

        if (m <= 1 || n <= 1) {
            return empty_out;
        }

        PRIMME_INT min_dim = std::min(m, n);
        PRIMME_INT k_eff = std::min<PRIMME_INT>(std::max<PRIMME_INT>(1, static_cast<PRIMME_INT>(k)), min_dim - 1);

        size_t m_sz = static_cast<size_t>(m);
        size_t n_sz = static_cast<size_t>(n);
        size_t k_sz = static_cast<size_t>(k_eff);
        if ((m_sz + n_sz) > 0 && k_sz > (std::numeric_limits<size_t>::max() / (m_sz + n_sz))) {
            throw std::overflow_error("PRIMME buffer size overflow");
        }

        if (verbose) {
            if (nnz > 0) {
                stdout_printf("PRIMME_SVDS (%s) -- A: %lld x %lld (nnz: %llu)\n",
                              label, static_cast<long long>(m), static_cast<long long>(n), nnz);
            }
            else {
                stdout_printf("PRIMME_SVDS (%s) -- A: %lld x %lld\n",
                              label, static_cast<long long>(m), static_cast<long long>(n));
            }
            FLUSH;
        }

        primme_svds_params primme_svds;
        primme_svds_initialize(&primme_svds);

        primme_svds.m = m;
        primme_svds.n = n;
        primme_svds.numSvals = k_eff;
        primme_svds.matrixMatvec = primmeMatvec;
        primme_svds.matrix = cb_ctx;

        primme_svds_set_method(primme_svds_default, PRIMME_DEFAULT_METHOD,
                               PRIMME_DEFAULT_METHOD, &primme_svds);
        primme_svds.eps = 1e-6;

        if (max_it > 0) {
            primme_svds.maxMatvecs = static_cast<PRIMME_INT>(max_it) * k_eff;
        } else {
            primme_svds.maxMatvecs = static_cast<PRIMME_INT>(1000) * k_eff;
        }

        // PRIMME requires iseed values in [0, 4095] with iseed[3] odd.
        // We derive them from the user seed via modular arithmetic.
        if (seed != 0) {
            unsigned int s = static_cast<unsigned int>(seed < 0 ? -seed : seed);
            primme_svds.iseed[0] = static_cast<PRIMME_INT>((s + 0) % 4096);
            primme_svds.iseed[1] = static_cast<PRIMME_INT>((s + 1) % 4096);
            primme_svds.iseed[2] = static_cast<PRIMME_INT>((s + 2) % 4096);
            // iseed[3] must be odd.
            PRIMME_INT s3 = static_cast<PRIMME_INT>((s + 3) % 4096);
            primme_svds.iseed[3] = (s3 % 2 == 0) ? (s3 + 1) % 4096 : s3;
        }

        primme_svds.printLevel = verbose ? 1 : 0;

        std::vector<double> svals(k_sz);
        std::vector<double> svecs((m_sz + n_sz) * k_sz);
        std::vector<double> rnorms(k_sz);

        int ret = dprimme_svds(svals.data(), svecs.data(), rnorms.data(), &primme_svds);
        if (ret != 0) {
            stderr_printf("PRIMME_SVDS returned error code: %d\n", ret);
            FLUSH;
            primme_svds_free(&primme_svds);
            return empty_out;
        }

        if (verbose) {
            stdout_printf("PRIMME_SVDS converged: %d singular values computed\n",
                          static_cast<int>(primme_svds.initSize));
            FLUSH;
        }

        actionet::SVDResult out;
        out.sigma = arma::vec(svals.data(), k_sz, true, true);
        out.U = arma::mat(m_sz, k_sz);
        out.V = arma::mat(n_sz, k_sz);

        for (size_t i = 0; i < k_sz; i++) {
            std::memcpy(out.U.colptr(i), svecs.data() + i * m_sz, m_sz * sizeof(double));
            std::memcpy(out.V.colptr(i), svecs.data() + k_sz * m_sz + i * n_sz, n_sz * sizeof(double));
        }

        primme_svds_free(&primme_svds);

        arma::field<arma::mat> oriented = actionet::svdFieldFromResult(out);
        actionet::orient_SVD(oriented);
        return actionet::svdResultFromField(oriented);
    }
} // namespace

namespace actionet {

arma::field<arma::mat> svdPRIMME(const arma::sp_mat& A, int k, int max_it, int seed, bool verbose) {
    PrimmeCallbackCtx ctx{static_cast<void*>(const_cast<arma::sp_mat*>(&A)), sparseDispatch, A.n_rows, A.n_cols, nullptr};
    actionet::SVDResult svd = runPrimmeCore(static_cast<PRIMME_INT>(A.n_rows), static_cast<PRIMME_INT>(A.n_cols),
                                            k, max_it, seed, verbose, &ctx, "sparse",
                                            static_cast<unsigned long long>(A.n_nonzero));
    return actionet::svdFieldFromResult(svd);
}

arma::field<arma::mat> svdPRIMME(const arma::mat& A, int k, int max_it, int seed, bool verbose) {
    PrimmeCallbackCtx ctx{static_cast<void*>(const_cast<arma::mat*>(&A)), denseDispatch, A.n_rows, A.n_cols, nullptr};
    actionet::SVDResult svd = runPrimmeCore(static_cast<PRIMME_INT>(A.n_rows), static_cast<PRIMME_INT>(A.n_cols),
                                            k, max_it, seed, verbose, &ctx, "dense");
    return actionet::svdFieldFromResult(svd);
}

SVDResult runSVD_PRIMME_Operator(const MatrixOperator& op, int k, int max_it, int seed, bool verbose) {
    PrimmeOperatorCtx op_ctx{&op};
    PrimmeCallbackCtx ctx{static_cast<void*>(&op_ctx), operatorDispatch, op.rows(), op.cols(), &op};
    return runPrimmeCore(static_cast<PRIMME_INT>(op.rows()), static_cast<PRIMME_INT>(op.cols()),
                         k, max_it, seed, verbose, &ctx, "operator");
}

} // namespace actionet
