#include "utils_internal/utils_stats.hpp"
#include "utils_internal/utils_parallel.hpp"

namespace {
    enum class ZScoreStrategy { Standard, Robust };

    arma::mat zscore_impl(arma::mat& A, int dim, int thread_no, ZScoreStrategy strategy) {
        int N = (dim == 0) ? A.n_cols : A.n_rows;

        int threads_use = actionet::get_num_threads_nested_safe(N, thread_no);
        #pragma omp parallel for num_threads(threads_use)
        for (size_t j = 0; j < N; j++) {
            arma::vec v = (dim == 0) ? arma::vec(A.col(j)) : arma::vec(A.row(j));

            double center, scale;
            if (strategy == ZScoreStrategy::Standard) {
                center = arma::mean(v);
                scale  = arma::stddev(v);
            } else {
                center = arma::median(v);
                scale  = arma::median(arma::abs(v - center));
            }

            arma::vec z = (v - center) / scale;
            if (dim == 0) {
                A.col(j) = z;
            } else {
                A.row(j) = z;
            }
        }

        A.replace(arma::datum::nan, 0);
        return A;
    }
} // anonymous namespace

namespace actionet {

arma::mat zscore(arma::mat& A, int dim, int thread_no) {
    return zscore_impl(A, dim, thread_no, ZScoreStrategy::Standard);
}

arma::mat robust_zscore(arma::mat& A, int dim, int thread_no) {
    return zscore_impl(A, dim, thread_no, ZScoreStrategy::Robust);
}

arma::mat tzscoret(arma::mat& A) {
    arma::mat At = A.t();
    A = zscore(At);
    return (A.t());
}

arma::mat mean_center(const arma::mat& A) {
    arma::mat A_centered = A;
    arma::rowvec mu = arma::rowvec(mean(A, 0));

    A_centered.each_row() -= mu;

    return A_centered;
}

} // namespace actionet
