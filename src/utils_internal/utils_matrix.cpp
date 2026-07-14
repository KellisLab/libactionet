// Sparse-dense matrix operations using native Armadillo
// Provides thread-safety and support for large matrices (>2^31 nnz)

#include "utils_internal/utils_matrix.hpp"
#include "utils_internal/utils_parallel.hpp"

namespace actionet {

// Thread-safe parallel sparse-dense matrix product
// Each thread independently computes a subset of output columns
arma::mat spmat_mat_product_parallel(const arma::sp_mat& A, const arma::mat& B, int thread_no) {
    if (A.n_cols != B.n_rows) {
        stderr_printf("spmat_mat_product_parallel:: Inner dimension of matrices should match\n");
        return arma::mat();
    }

    size_t N = B.n_cols;
    arma::mat res(A.n_rows, N);

    int threads_use = get_num_threads_nested_safe(N, thread_no);

    #pragma omp parallel for num_threads(threads_use)
    for (size_t j = 0; j < N; ++j) {
        res.col(j) = A * B.col(j);
    }

    return res;
}

} // namespace actionet
