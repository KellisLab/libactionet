// Singular value decomposition (SVD) using PRIMME_SVDS
#include "decomposition/svd_primme.hpp"
#include "utils_internal/utils_decomp.hpp"
#include "blas_deps.hpp"
#include "primme.h"
#include "primme_svds.h"
#include <cstring>

// Matrix-vector product callback for PRIMME (sparse matrix)
static void sparseMatvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy, int* blockSize,
                         int* trans, primme_svds_params* primme_svds, int* err) {
    arma::sp_mat* A = static_cast<arma::sp_mat*>(primme_svds->matrix);
    double* xvec = static_cast<double*>(x);
    double* yvec = static_cast<double*>(y);

    int m = A->n_rows;
    int n = A->n_cols;

    for (int i = 0; i < *blockSize; i++) {
        arma::vec x_col(xvec + (*ldx) * i, (*trans == 0) ? n : m, false, true);
        arma::vec y_col(yvec + (*ldy) * i, (*trans == 0) ? m : n, false, true);

        if (*trans == 0) {
            // y = A * x
            y_col = (*A) * x_col;
        } else {
            // y = A' * x
            y_col = A->t() * x_col;
        }
    }

    *err = 0;
}

// Matrix-vector product callback for PRIMME (dense matrix)
static void denseMatvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy, int* blockSize,
                        int* trans, primme_svds_params* primme_svds, int* err) {
    arma::mat* A = static_cast<arma::mat*>(primme_svds->matrix);
    double* xvec = static_cast<double*>(x);
    double* yvec = static_cast<double*>(y);

    int m = A->n_rows;
    int n = A->n_cols;

    for (int i = 0; i < *blockSize; i++) {
        arma::vec x_col(xvec + (*ldx) * i, (*trans == 0) ? n : m, false, true);
        arma::vec y_col(yvec + (*ldy) * i, (*trans == 0) ? m : n, false, true);

        if (*trans == 0) {
            // y = A * x
            y_col = (*A) * x_col;
        } else {
            // y = A' * x
            y_col = A->t() * x_col;
        }
    }

    *err = 0;
}

arma::field<arma::mat> svdPRIMME(arma::sp_mat& A, int k, int max_it, int seed, bool verbose) {
    PRIMME_INT m = A.n_rows;
    PRIMME_INT n = A.n_cols;

    k = std::min(k, std::min((int)m, (int)n) - 1);

    if (verbose) {
        stdout_printf("PRIMME_SVDS (sparse) -- A: %lld x %lld (nnz: %llu)\n",
                     (long long)m, (long long)n, (unsigned long long)A.n_nonzero);
        FLUSH;
    }

    // Initialize PRIMME_SVDS parameters
    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);

    // Set problem dimensions
    primme_svds.m = m;
    primme_svds.n = n;
    primme_svds.numSvals = k;

    // Set matrix-vector product callback
    primme_svds.matrixMatvec = sparseMatvec;
    primme_svds.matrix = &A;

    // Set method (primme_svds_default uses a good general-purpose method)
    primme_svds_set_method(primme_svds_default, PRIMME_DEFAULT_METHOD,
                           PRIMME_DEFAULT_METHOD, &primme_svds);

    // Set convergence tolerance
    primme_svds.eps = 1e-6;

    // Set maximum iterations
    if (max_it > 0) {
        primme_svds.maxMatvecs = max_it * k;
    }

    // Set random seed for reproducibility
    if (seed != 0) {
        primme_svds.iseed[0] = seed;
        primme_svds.iseed[1] = seed + 1;
        primme_svds.iseed[2] = seed + 2;
        primme_svds.iseed[3] = seed + 3;
    }

    // Control output verbosity
    if (verbose) {
        primme_svds.printLevel = 1;
    } else {
        primme_svds.printLevel = 0;
    }

    // Allocate result arrays
    double* svals = new double[k];
    double* svecs = new double[(m + n) * k];
    double* rnorms = new double[k];

    // Call PRIMME_SVDS solver
    int ret = dprimme_svds(svals, svecs, rnorms, &primme_svds);

    if (ret != 0) {
        stderr_printf("PRIMME_SVDS returned error code: %d\n", ret);
        delete[] svals;
        delete[] svecs;
        delete[] rnorms;
        primme_svds_free(&primme_svds);

        // Return empty result on error
        arma::field<arma::mat> out(3);
        out(0) = arma::mat();
        out(1) = arma::mat();
        out(2) = arma::mat();
        return out;
    }

    if (verbose) {
        stdout_printf("PRIMME_SVDS converged: %d singular values computed\n",
                     (int)primme_svds.initSize);
        FLUSH;
    }

    // Extract results
    arma::vec S(svals, k, true, true);  // Singular values

    // U matrix (left singular vectors, m × k)
    arma::mat U(m, k);
    for (int i = 0; i < k; i++) {
        std::memcpy(U.colptr(i), svecs + i * m, m * sizeof(double));
    }

    // V matrix (right singular vectors, n × k)
    arma::mat V(n, k);
    for (int i = 0; i < k; i++) {
        std::memcpy(V.colptr(i), svecs + k * m + i * n, n * sizeof(double));
    }

    // Clean up
    delete[] svals;
    delete[] svecs;
    delete[] rnorms;
    primme_svds_free(&primme_svds);

    // Return results in field
    arma::field<arma::mat> out(3);
    out(0) = U;
    out(1) = arma::diagmat(S);  // Return diagonal matrix for compatibility
    out(2) = V;

    return out;
}

arma::field<arma::mat> svdPRIMME(arma::mat& A, int k, int max_it, int seed, bool verbose) {
    PRIMME_INT m = A.n_rows;
    PRIMME_INT n = A.n_cols;

    k = std::min(k, std::min((int)m, (int)n) - 1);

    if (verbose) {
        stdout_printf("PRIMME_SVDS (dense) -- A: %lld x %lld\n",
                     (long long)m, (long long)n);
        FLUSH;
    }

    // Initialize PRIMME_SVDS parameters
    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);

    // Set problem dimensions
    primme_svds.m = m;
    primme_svds.n = n;
    primme_svds.numSvals = k;

    // Set matrix-vector product callback
    primme_svds.matrixMatvec = denseMatvec;
    primme_svds.matrix = &A;

    // Set method
    primme_svds_set_method(primme_svds_default, PRIMME_DEFAULT_METHOD,
                           PRIMME_DEFAULT_METHOD, &primme_svds);

    // Set convergence tolerance
    primme_svds.eps = 1e-6;

    // Set maximum iterations
    if (max_it > 0) {
        primme_svds.maxMatvecs = max_it * k;
    }

    // Set random seed
    if (seed != 0) {
        primme_svds.iseed[0] = seed;
        primme_svds.iseed[1] = seed + 1;
        primme_svds.iseed[2] = seed + 2;
        primme_svds.iseed[3] = seed + 3;
    }

    // Control output verbosity
    if (verbose) {
        primme_svds.printLevel = 1;
    } else {
        primme_svds.printLevel = 0;
    }

    // Allocate result arrays
    double* svals = new double[k];
    double* svecs = new double[(m + n) * k];
    double* rnorms = new double[k];

    // Call PRIMME_SVDS solver
    int ret = dprimme_svds(svals, svecs, rnorms, &primme_svds);

    if (ret != 0) {
        stderr_printf("PRIMME_SVDS returned error code: %d\n", ret);
        delete[] svals;
        delete[] svecs;
        delete[] rnorms;
        primme_svds_free(&primme_svds);

        // Return empty result on error
        arma::field<arma::mat> out(3);
        out(0) = arma::mat();
        out(1) = arma::mat();
        out(2) = arma::mat();
        return out;
    }

    if (verbose) {
        stdout_printf("PRIMME_SVDS converged: %d singular values computed\n",
                     (int)primme_svds.initSize);
        FLUSH;
    }

    // Extract results
    arma::vec S(svals, k, true, true);

    // U matrix (left singular vectors)
    arma::mat U(m, k);
    for (int i = 0; i < k; i++) {
        std::memcpy(U.colptr(i), svecs + i * m, m * sizeof(double));
    }

    // V matrix (right singular vectors)
    arma::mat V(n, k);
    for (int i = 0; i < k; i++) {
        std::memcpy(V.colptr(i), svecs + k * m + i * n, n * sizeof(double));
    }

    // Clean up
    delete[] svals;
    delete[] svecs;
    delete[] rnorms;
    primme_svds_free(&primme_svds);

    // Return results
    arma::field<arma::mat> out(3);
    out(0) = U;
    out(1) = arma::diagmat(S);
    out(2) = V;

    return out;
}
