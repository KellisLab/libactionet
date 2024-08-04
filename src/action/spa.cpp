// Solves separable NMF problem
#include "spa.hpp"

namespace ACTIONet
{

    SPA_results run_SPA(arma::mat &A, int k)
    {
        SPA_results res;

        int n = A.n_cols;
        arma::uvec K(k); // selected columns from A
        K.zeros();

        arma::rowvec normM = sum(square(A), 0);
        arma::rowvec normM1 = normM;

        arma::mat U(A.n_rows, k);

        arma::vec norm_trace = arma::zeros(k);
        double eps = 1e-16;

        for (int i = 1; i <= k; i++)
        {
            // Find the column with maximum norm. In case of having more than one column
            // with almost very small diff in norm, pick the one that originally had the
            // largest norm
            double a = max(normM);
            norm_trace(i - 1) = a;

            arma::uvec b = arma::find((a * arma::ones(1, n) - normM) / a <= eps);
            if (b.n_elem == 0)
            {
                break;
            }
            else if (b.n_elem > 1)
            {
                arma::uword idx = arma::index_max(normM1(b));
                K(i - 1) = b(idx);
            }
            else
            {
                K(i - 1) = b(0);
            }

            // Pick column
            U.col(i - 1) = A.col(K(i - 1));

            // Orthogonalize with respect to current basis
            if (i > 1)
            {
                for (int j = 1; j <= i - 1; j++)
                {
                    U.col(i - 1) =
                        U.col(i - 1) - sum(U.col(j - 1) % U.col(i - 1)) * U.col(j - 1);
                }
            }
            double nm = norm(U.col(i - 1), 2);
            if (nm > 0)
                U.col(i - 1) /= nm;

            // Update column norms
            arma::vec u = U.col(i - 1);
            if (i > 1)
            {
                for (int j = i - 1; 1 <= j; j--)
                {
                    u = u - arma::sum(U.col(j - 1) % u) * U.col(j - 1);
                }
            }
            normM = normM - arma::square(u.t() * A);
            normM.transform([](double val)
                            { return (val < 0 ? 0 : val); });
        }

        res.selected_columns = K;
        res.column_norms = norm_trace;

        return res;
    }

} // namespace ACTIONet

// SPA_results run_SPA_rows_sparse(arma::sp_mat &A, int k)
// {
//     int m = A.n_rows;
//     int n = A.n_cols;
//     arma::sp_mat A_sq = square(A);

//     cholmod_common chol_c;
//     cholmod_start(&chol_c);
//     chol_c.final_ll = 1; /* LL' form of simplicial factorization */

//     cholmod_sparse *AS = as_cholmod_sparse(A, AS, &chol_c);
//     cholmod_sparse *AS_sq = as_cholmod_sparse(A_sq, AS_sq, &chol_c);

//     SPA_results res;

//     arma::uvec K(k); // selected columns from A

//     arma::vec o = arma::ones(n);
//     arma::vec normM(m);
//     dsdmult('n', m, n, AS_sq, o.memptr(), normM.memptr(), &chol_c);
//     arma::vec normM1 = normM;
//     arma::mat U(n, k);

//     arma::vec norm_trace = arma::zeros(k);
//     double eps = 1e-6;
//     for (int i = 0; i < k; i++)
//     {
//         // Find the column with maximum norm. In case of having more than one column
//         // with almost very small diff in norm, pick the one that originally had the
//         // largest norm
//         double a = arma::max(normM);
//         norm_trace(i) = a;

//         arma::uvec b = arma::find((a * arma::ones(m, 1) - normM) / a <= eps);

//         if (b.n_elem > 1)
//         {
//             arma::uword idx = arma::index_max(normM1(b));
//             K(i) = b(idx);
//         }
//         else
//         {
//             K(i) = b(0);
//         }

//         // Pick row
//         U.col(i) = arma::vec(trans(A.row(K(i))));

//         // Orthogonalize with respect to current basis
//         for (int j = 0; j < i - 1; j++)
//         {
//             U.col(i) = U.col(i) - dot(U.col(j), U.col(i)) * U.col(j);
//         }
//         U.col(i) = U.col(i) / norm(U.col(i), 2);

//         // Update column norms
//         arma::vec u = U.col(i);
//         for (int j = i - 1; 0 <= j; j--)
//         {
//             u = u - arma::dot(U.col(j), u) * U.col(j);
//         }
//         arma::vec r(m);
//         dsdmult('n', m, n, AS, u.memptr(), r.memptr(), &chol_c);

//         arma::uvec idx = find(U > 0);
//         double perc = 100 * idx.n_elem / U.n_elem;
//         stdout_printf("\t%d- res_norm = %f, U_density = %.2f%% (%d nnz)\n", i, a,
//                       perc, idx.n_elem);
//         FLUSH;

//         normM = normM - (r % r);
//     }

//     res.selected_columns = K;
//     res.column_norms = norm_trace;

//     cholmod_free_sparse(&AS, &chol_c);
//     cholmod_free_sparse(&AS_sq, &chol_c);
//     cholmod_finish(&chol_c);

//     return res;
// }
