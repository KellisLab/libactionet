// Successive projection algorithm (SPA)
#include "action/spa.hpp"

namespace ACTIONet {

    SPA_results run_SPA(arma::mat &A, int k) {
        SPA_results res;

        int n = A.n_cols;
        arma::uvec K(k); // selected columns from A
        K.zeros();

        arma::rowvec normM = arma::sum(arma::square(A), 0);
        arma::rowvec normM1 = normM;

        arma::mat U(A.n_rows, k);

        arma::vec norm_trace = arma::zeros(k);
        double eps = 1e-16;

        for (int i = 1; i <= k; i++) {
            // Find the column with maximum norm. In case of having more than one column
            // with almost very small diff in norm, pick the one that originally had the
            // largest norm
            double a = arma::max(normM);
            norm_trace(i - 1) = a;

            arma::uvec b = arma::find((a * arma::ones(1, n) - normM) / a <= eps);
            if (b.n_elem == 0) {
                break;
            } else if (b.n_elem > 1) {
                arma::uword idx = arma::index_max(normM1(b));
                K(i - 1) = b(idx);
            } else {
                K(i - 1) = b(0);
            }

            // Pick column
            U.col(i - 1) = A.col(K(i - 1));

            // Orthogonalize with respect to current basis
            if (i > 1) {
                for (int j = 1; j <= i - 1; j++) {
                    U.col(i - 1) =
                            U.col(i - 1) - sum(U.col(j - 1) % U.col(i - 1)) * U.col(j - 1);
                }
            }
            double nm = arma::norm(U.col(i - 1), 2);
            if (nm > 0)
                U.col(i - 1) /= nm;

            // Update column norms
            arma::vec u = U.col(i - 1);
            if (i > 1) {
                for (int j = i - 1; 1 <= j; j--) {
                    u = u - arma::sum(U.col(j - 1) % u) * U.col(j - 1);
                }
            }
            normM = normM - arma::square(u.t() * A);
            normM.transform([](double val) { return (val < 0 ? 0 : val); });
        }

        res.selected_columns = K;
        res.column_norms = norm_trace;

        return res;
    }

} // namespace ACTIONet
