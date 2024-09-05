#include "utils_internal/utils_matrix.hpp"
#include "utils_internal/utils_parallel.hpp"

// Mat-vec product (Ax)
void dsdmult(char transpose, int n_rows, int n_cols, const void* A, const double* x, double* out,
             cholmod_common* chol_cp) {
    int t = transpose == 't' ? 1 : 0; // 'n': computes Ax, 't': computes A'x
    auto* cha = (cholmod_sparse*)A;

    // x
    cholmod_dense chb;
    chb.nrow = t ? n_rows : n_cols;
    chb.d = chb.nrow;
    chb.ncol = 1;
    chb.nzmax = chb.nrow;
    chb.xtype = cha->xtype;
    chb.dtype = 0;
    chb.x = (void*)x;
    chb.z = (void*)NULL;

    // out
    cholmod_dense chc;
    chc.nrow = t ? n_cols : n_rows;
    chc.d = chc.nrow;
    chc.ncol = 1;
    chc.nzmax = chc.nrow;
    chc.xtype = cha->xtype;
    chc.dtype = 0;
    chc.x = (void*)out;
    chc.z = (void*)NULL;

    double one[] = {1, 0}, zero[] = {0, 0};

    cholmod_sdmult(cha, t, one, zero, &chb, &chc, chol_cp);
}

cholmod_sparse* as_cholmod_sparse(const arma::sp_mat& A, cholmod_sparse* chol_A, cholmod_common* chol_c) {
    int nrow = A.n_rows, ncol = A.n_cols, nz = A.n_nonzero;
    cholmod_allocate_work(0, std::max(nrow, ncol), 0, chol_c);
    chol_A = cholmod_allocate_sparse(nrow, ncol, nz, 1 /*sorted*/, 1 /*packed*/, 0 /*NOT symmetric*/, CHOLMOD_REAL,
                                     chol_c);

    int* ptr = (int*)chol_A->p;
    double* x_ptr = (double*)chol_A->x;
    int* i_ptr = (int*)chol_A->i;

    A.sync();
    {
        for (int k = 0; k < A.n_nonzero; k++) {
            x_ptr[k] = (A.values)[k];
            i_ptr[k] = (A.row_indices)[k];
        }
        for (int k = 0; k < A.n_cols + 1; k++) {
            ptr[k] = A.col_ptrs[k];
        }
    }

    return chol_A;
}

arma::sp_mat& as_arma_sparse(const cholmod_sparse* chol_A, arma::sp_mat& A, cholmod_common* chol_c) {
    // Allocate space
    A = arma::sp_mat(chol_A->nrow, chol_A->ncol);
    A.mem_resize(static_cast<unsigned>(chol_A->nzmax));

    A.sync();
    double* in_x_ptr = (double*)chol_A->x;
    int* in_i_ptr = (int*)chol_A->i;
    double* out_x_ptr = (double*)arma::access::rwp(A.values);
    arma::uword* out_i_ptr = arma::access::rwp(A.row_indices);
    for (int k = 0; k < chol_A->nzmax; k++) {
        out_x_ptr[k] = in_x_ptr[k];
        out_i_ptr[k] = in_i_ptr[k];
    }

    int* in_p_ptr = (int*)chol_A->p;
    arma::uword* out_p_ptr = arma::access::rwp(A.col_ptrs);
    for (int k = 0; k < chol_A->ncol; k++) {
        out_p_ptr[k] = in_p_ptr[k];
    }

    // important: set the sentinel as well
    arma::access::rwp(A.col_ptrs)[chol_A->ncol] = chol_A->nzmax;

    // set the number of non-zero elements
    arma::access::rw(A.n_nonzero) = chol_A->nzmax;

    return A;
}

arma::vec spmat_vec_product(const arma::sp_mat& A, arma::vec& x) {
    arma::mat X = arma::mat(x);
    arma::mat Ax = spmat_mat_product(A, X);

    return (Ax.col(0));
}

arma::mat spmat_mat_product(const arma::sp_mat& A, arma::mat& B) {
    if (A.n_cols != B.n_rows) {
        stderr_printf("spmat_mat_product:: Inner dimension of matrices should match\n.");
        return (arma::mat());
    }

    cholmod_common chol_c;
    cholmod_start(&chol_c);

    cholmod_sparse* chol_A;
    chol_A = as_cholmod_sparse(A, chol_A, &chol_c);

    cholmod_dense* chol_B = cholmod_allocate_dense(B.n_rows, B.n_cols, B.n_rows, CHOLMOD_REAL, &chol_c);
    chol_B->x = (void*)B.memptr();
    chol_B->z = (void*)NULL;

    arma::mat res = arma::zeros(A.n_rows, B.n_cols);
    cholmod_dense* out = cholmod_allocate_dense(A.n_rows, B.n_cols, A.n_rows, CHOLMOD_REAL, &chol_c);
    out->x = (void*)res.memptr();
    out->z = (void*)NULL;

    double one[] = {1, 0}, zero[] = {0, 0};
    cholmod_sdmult(chol_A, 0, one, zero, chol_B, out, &chol_c);

    cholmod_free_sparse(&chol_A, &chol_c);
    cholmod_finish(&chol_c);
    return (res);
}

// TODO: REMOVE?
arma::sp_mat spmat_spmat_product(const arma::sp_mat& A, const arma::sp_mat& B) {
    arma::sp_mat res;

    if (A.n_cols != B.n_rows) {
        stderr_printf("spmat_spmat_product:: Inner dimension of matrices should match\n.");
        return (res);
    }
    cholmod_common chol_c;
    cholmod_start(&chol_c);

    cholmod_sparse *chol_A, *chol_B;
    chol_A = as_cholmod_sparse(A, chol_A, &chol_c);
    chol_B = as_cholmod_sparse(B, chol_B, &chol_c);

    cholmod_sparse* chol_res = cholmod_ssmult(chol_A, chol_B, 0, true,
                                              true, &chol_c);

    res = as_arma_sparse(chol_res, res, &chol_c);

    cholmod_free_sparse(&chol_A, &chol_c);
    cholmod_free_sparse(&chol_B, &chol_c);
    cholmod_finish(&chol_c);

    return (res);
}

arma::mat spmat_mat_product_parallel(const arma::sp_mat& A, arma::mat& B, int thread_no) {
    if (A.n_cols != B.n_rows) {
        stderr_printf("spmat_mat_product_parallel:: Inner dimension of matrices should match\n.");
        return (arma::mat());
    }

    cholmod_common chol_c;
    cholmod_start(&chol_c);

    cholmod_sparse* chol_A;
    chol_A = as_cholmod_sparse(A, chol_A, &chol_c);

    int M = A.n_rows;
    int N = B.n_cols;
    arma::mat res = arma::zeros(M, N);

    int threads_use = get_num_threads(N, thread_no);
    int slice_size = std::ceil((double)N / threads_use);

    #pragma omp parallel for num_threads(threads_use)
    for (unsigned int k = 0; k < N; k++) {
        int i = k * slice_size;
        if (i <= (N - 1)) {
            int j = (k + 1) * slice_size - 1;
            if (j > (N - 1))
                j = N - 1;

            arma::mat subB = B.cols(i, j);

            // Magic starts here!
            cholmod_dense* chol_B = cholmod_allocate_dense(subB.n_rows, subB.n_cols, subB.n_rows, CHOLMOD_REAL,
                                                           &chol_c);
            chol_B->x = (void*)subB.memptr();
            chol_B->z = (void*)NULL;

            arma::mat subC = arma::zeros(A.n_rows, subB.n_cols);
            cholmod_dense* out = cholmod_allocate_dense(A.n_rows, subB.n_cols, A.n_rows, CHOLMOD_REAL, &chol_c);
            out->x = (void*)subC.memptr();
            out->z = (void*)NULL;

            double one[] = {1, 0}, zero[] = {0, 0};
            cholmod_sdmult(chol_A, 0, one, zero, chol_B, out, &chol_c);

            res.cols(i, j) = subC;
        }
    }

    cholmod_free_sparse(&chol_A, &chol_c);
    cholmod_finish(&chol_c);
    return (res);
}

arma::mat mat_mat_product_parallel(const arma::mat& A, arma::mat& B, int thread_no) {
    if (thread_no <= 0) {
        thread_no = SYS_THREADS_DEF;
    }

    int M = A.n_rows;
    int N = B.n_cols;
    arma::mat res = arma::zeros(M, N);

    if (thread_no > N) {
        thread_no = N;
        mini_thread::parallelFor(
            0, thread_no, [&](size_t k) {
                arma::vec u = B.col(k);
                arma::vec v = A * u;
                res.col(k) = v;
            },
            thread_no);
    }
    else {
        int slice_size = std::ceil((double)N / thread_no);

        mini_thread::parallelFor(
            0, thread_no, [&](size_t k) {
                int i = k * slice_size;
                if (i <= (N - 1)) {
                    int j = (k + 1) * slice_size - 1;
                    if (j > (N - 1))
                        j = N - 1;

                    arma::mat subB = B.cols(i, j);
                    arma::mat subC = A * subB;
                    res.cols(i, j) = subC;
                }
            },
            thread_no);
    }

    return (res);
}
