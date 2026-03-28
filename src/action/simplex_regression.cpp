// Simplex regression algorithm
#include "action/simplex_regression.hpp"
#include "utils_internal/utils_active_set.hpp"
#include "utils_internal/utils_parallel.hpp"

namespace actionet {

    arma::mat runSimplexRegression(const arma::mat &A, const arma::mat &B, bool computeXtX) {

        double lambda2 = 1e-5, epsilon = 1e-5;

        int ncols = static_cast<int>(B.n_cols);
        arma::mat X = arma::zeros(A.n_cols, ncols);

        int nthreads = omp_in_parallel() ? 1 : get_num_threads(ncols);

        if (computeXtX) {
            double lam2sq = lambda2 * lambda2;
            arma::mat G = arma::trans(A) * A + lam2sq;
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

        X = arma::clamp(X, 0, 1);
        X = arma::normalise(X, 1);

        return (X);
    }

} // namespace actionet
