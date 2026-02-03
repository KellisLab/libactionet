#include "utils_internal/utils_matrix.hpp"
#include "utils_internal/utils_parallel.hpp"

// Sparse matrix-dense vector product: y = A*x or y = A'*x
arma::vec spmat_vec_product(const arma::sp_mat& A, const arma::vec& x) {
    return A * x;
}

// Sparse matrix-dense vector product with transpose option
arma::vec spmat_vec_product_t(const arma::sp_mat& A, const arma::vec& x, bool transpose) {
    if (transpose) {
        return A.t() * x;
    } else {
        return A * x;
    }
}

// Sparse matrix-dense matrix product: C = A*B
arma::mat spmat_mat_product(const arma::sp_mat& A, const arma::mat& B) {
    if (A.n_cols != B.n_rows) {
        stderr_printf("spmat_mat_product:: Inner dimension of matrices should match\n");
        return arma::mat();
    }

    return A * B;
}

// Thread-safe parallel sparse-dense matrix product
// Each thread independently computes a subset of output columns
arma::mat spmat_mat_product_parallel(const arma::sp_mat& A, const arma::mat& B, int thread_no) {
    if (A.n_cols != B.n_rows) {
        stderr_printf("spmat_mat_product_parallel:: Inner dimension of matrices should match\n");
        return arma::mat();
    }

    size_t M = A.n_rows;
    size_t N = B.n_cols;
    arma::mat res(M, N);

    int threads_use = get_num_threads(N, thread_no);

    #pragma omp parallel for num_threads(threads_use)
    for (size_t j = 0; j < N; ++j) {
        res.col(j) = A * B.col(j);
    }

    return res;
}

// Sparse-sparse matrix product: C = A*B
arma::sp_mat spmat_spmat_product(const arma::sp_mat& A, const arma::sp_mat& B) {
    if (A.n_cols != B.n_rows) {
        stderr_printf("spmat_spmat_product:: Inner dimension of matrices should match\n");
        return arma::sp_mat();
    }

    return A * B;
}
