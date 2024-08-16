// Simplex regression algorithm
#include "action/simplex_regression.hpp"
#include "utils_internal/utils_active_set.hpp"

namespace ACTIONet {

    arma::mat run_simplex_regression(arma::mat &A, arma::mat &B, bool computeXtX) {

        double lambda2 = 1e-5, epsilon = 1e-5;

        arma::mat X = arma::zeros(A.n_cols, B.n_cols);
        if (computeXtX) {
            double lam2sq = lambda2 * lambda2;
            arma::mat G = arma::trans(A) * A + lam2sq;
            for (int i = 0; i < B.n_cols; i++) {
                arma::vec b = B.col(i);
                X.col(i) = activeSetS_arma(A, b, G, lambda2, epsilon);
            }
        } else {
            for (int i = 0; i < B.n_cols; i++) {
                arma::vec b = B.col(i);
                X.col(i) = activeSet_arma(A, b, lambda2, epsilon);
            }
        }

        X = arma::clamp(X, 0, 1);
        X = arma::normalise(X, 1);

        return (X);
    }

} // namespace ACTIONet
