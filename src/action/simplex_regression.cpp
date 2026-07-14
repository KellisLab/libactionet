// Simplex regression algorithm
#include "action/simplex_regression.hpp"
#include "utils_internal/utils_active_set.hpp"
#include "utils_internal/utils_action_numeric_policy.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_small_dense.hpp"

namespace actionet {

    namespace numeric_policy = utils_internal::action_numeric_policy;

    arma::mat runSimplexRegression(const arma::mat &A, const arma::mat &B, bool computeXtX) {

        constexpr double lambda2 = numeric_policy::simplex_l2_regularization;
        constexpr double epsilon = numeric_policy::simplex_optimality_tolerance;

        int ncols = static_cast<int>(B.n_cols);
        arma::mat X = arma::zeros(A.n_cols, ncols);

        int nthreads = get_num_threads_nested_safe(ncols);

        if (computeXtX) {
            double lam2sq = lambda2 * lambda2;
            arma::mat G = utils_internal::small_dense::gram(A, lam2sq);
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (int i = 0; i < ncols; i++) {
                arma::vec b = B.col(i);
                X.col(i) = activeSetS_arma(A, b, G, lambda2, epsilon);
            }
        } else {
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (int i = 0; i < ncols; i++) {
                arma::vec b = B.col(i);
                X.col(i) = activeSet_arma(A, b, lambda2, epsilon);
            }
        }

        utils_internal::small_dense::clamp_and_normalize_columns(X);

        return (X);
    }

} // namespace actionet
