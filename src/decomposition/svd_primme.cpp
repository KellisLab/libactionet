// Singular value decomposition (SVD) using PRIMME_SVDS
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
    struct PrimmeOperatorCtx {
        const actionet::MatrixOperator* op;
    };

    static void sparseMatvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy, int* blockSize,
                             int* trans, primme_svds_params* primme_svds, int* err) {
        arma::sp_mat* A = static_cast<arma::sp_mat*>(primme_svds->matrix);
        double* xvec = static_cast<double*>(x);
        double* yvec = static_cast<double*>(y);

        arma::uword m = A->n_rows;
        arma::uword n = A->n_cols;

        for (int i = 0; i < *blockSize; i++) {
            arma::uword x_len = (*trans == 0) ? n : m;
            arma::uword y_len = (*trans == 0) ? m : n;

            arma::vec x_col(xvec + (*ldx) * i, x_len, false, true);
            arma::vec y_col(yvec + (*ldy) * i, y_len, false, true);

            if (*trans == 0) {
                y_col = (*A) * x_col;
            }
            else {
                y_col = A->t() * x_col;
            }
        }

        *err = 0;
    }

    static void denseMatvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy, int* blockSize,
                            int* trans, primme_svds_params* primme_svds, int* err) {
        arma::mat* A = static_cast<arma::mat*>(primme_svds->matrix);
        double* xvec = static_cast<double*>(x);
        double* yvec = static_cast<double*>(y);

        arma::uword m = A->n_rows;
        arma::uword n = A->n_cols;

        for (int i = 0; i < *blockSize; i++) {
            arma::uword x_len = (*trans == 0) ? n : m;
            arma::uword y_len = (*trans == 0) ? m : n;

            arma::vec x_col(xvec + (*ldx) * i, x_len, false, true);
            arma::vec y_col(yvec + (*ldy) * i, y_len, false, true);

            if (*trans == 0) {
                y_col = (*A) * x_col;
            }
            else {
                y_col = A->t() * x_col;
            }
        }

        *err = 0;
    }

    static void operatorMatvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy, int* blockSize,
                               int* trans, primme_svds_params* primme_svds, int* err) {
        PrimmeOperatorCtx* ctx = static_cast<PrimmeOperatorCtx*>(primme_svds->matrix);
        const actionet::MatrixOperator& op = *(ctx->op);

        double* xvec = static_cast<double*>(x);
        double* yvec = static_cast<double*>(y);

        arma::uword m = op.rows();
        arma::uword n = op.cols();

        for (int i = 0; i < *blockSize; i++) {
            arma::uword x_len = (*trans == 0) ? n : m;
            arma::uword y_len = (*trans == 0) ? m : n;

            arma::vec x_col(xvec + (*ldx) * i, x_len, false, true);
            arma::vec y_col(yvec + (*ldy) * i, y_len, false, true);

            if (*trans == 0) {
                op.matvec(x_col, y_col);
            }
            else {
                op.rmatvec(x_col, y_col);
            }
        }

        *err = 0;
    }

    actionet::SVDResult runPrimmeCore(PRIMME_INT m, PRIMME_INT n, int k, int max_it, int seed, bool verbose,
                                      void* matrix_ptr,
                                      void (*matvec_fn)(void*, PRIMME_INT*, void*, PRIMME_INT*, int*, int*,
                                                        primme_svds_params*, int*),
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
        primme_svds.matrixMatvec = matvec_fn;
        primme_svds.matrix = matrix_ptr;

        primme_svds_set_method(primme_svds_default, PRIMME_DEFAULT_METHOD,
                               PRIMME_DEFAULT_METHOD, &primme_svds);
        primme_svds.eps = 1e-6;

        if (max_it > 0) {
            primme_svds.maxMatvecs = static_cast<PRIMME_INT>(max_it) * k_eff;
        }

        if (seed != 0) {
            primme_svds.iseed[0] = seed;
            primme_svds.iseed[1] = seed + 1;
            primme_svds.iseed[2] = seed + 2;
            primme_svds.iseed[3] = seed + 3;
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

        arma::field<arma::mat> oriented = orient_SVD(actionet::svdFieldFromResult(out));
        return actionet::svdResultFromField(oriented);
    }
} // namespace

arma::field<arma::mat> svdPRIMME(arma::sp_mat& A, int k, int max_it, int seed, bool verbose) {
    actionet::SVDResult svd = runPrimmeCore(static_cast<PRIMME_INT>(A.n_rows), static_cast<PRIMME_INT>(A.n_cols),
                                            k, max_it, seed, verbose, &A, sparseMatvec, "sparse",
                                            static_cast<unsigned long long>(A.n_nonzero));
    return actionet::svdFieldFromResult(svd);
}

arma::field<arma::mat> svdPRIMME(arma::mat& A, int k, int max_it, int seed, bool verbose) {
    actionet::SVDResult svd = runPrimmeCore(static_cast<PRIMME_INT>(A.n_rows), static_cast<PRIMME_INT>(A.n_cols),
                                            k, max_it, seed, verbose, &A, denseMatvec, "dense");
    return actionet::svdFieldFromResult(svd);
}

namespace actionet {
    SVDResult runSVD_PRIMME_Operator(const MatrixOperator& op, int k, int max_it, int seed, bool verbose) {
        PrimmeOperatorCtx ctx{&op};
        return runPrimmeCore(static_cast<PRIMME_INT>(op.rows()), static_cast<PRIMME_INT>(op.cols()),
                             k, max_it, seed, verbose, &ctx, operatorMatvec, "operator");
    }
} // namespace actionet
