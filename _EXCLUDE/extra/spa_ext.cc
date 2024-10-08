#include "spa_ext.h"

actionet::ResSPA runSPA_rows_sparse(arma::sp_mat &A, int k) {
    int m = A.n_rows;
    int n = A.n_cols;
    arma::sp_mat A_sq = arma::square(A);

    cholmod_common chol_c;
    cholmod_start(&chol_c);
    chol_c.final_ll = 1; /* LL' form of simplicial factorization */

    cholmod_sparse *AS = as_cholmod_sparse(A, AS, &chol_c);
    cholmod_sparse *AS_sq = as_cholmod_sparse(A_sq, AS_sq, &chol_c);

    actionet::ResSPA res;

    arma::uvec K(k); // selected columns from A

    arma::vec o = arma::ones(n);
    arma::vec normM(m);
    dsdmult('n', m, n, AS_sq, o.memptr(), normM.memptr(), &chol_c);
    arma::vec normM1 = normM;
    arma::mat U(n, k);

    arma::vec norm_trace = arma::zeros(k);
    double eps = 1e-6;
    for (int i = 0; i < k; i++) {
        // Find the column with maximum norm. In case of having more than one column
        // with almost very small diff in norm, pick the one that originally had the
        // largest norm
        double a = arma::max(normM);
        norm_trace(i) = a;

        arma::uvec b = arma::find((a * arma::ones(m, 1) - normM) / a <= eps);

        if (b.n_elem > 1) {
            arma::uword idx = arma::index_max(normM1(b));
            K(i) = b(idx);
        } else {
            K(i) = b(0);
        }

        // Pick row
        U.col(i) = arma::vec(arma::trans(A.row(K(i))));

        // Orthogonalize with respect to current basis
        for (int j = 0; j < i - 1; j++) {
            U.col(i) = U.col(i) - arma::dot(U.col(j), U.col(i)) * U.col(j);
        }
        U.col(i) = U.col(i) / norm(U.col(i), 2);

        // Update column norms
        arma::vec u = U.col(i);
        for (int j = i - 1; 0 <= j; j--) {
            u = u - arma::dot(U.col(j), u) * U.col(j);
        }
        arma::vec r(m);
        dsdmult('n', m, n, AS, u.memptr(), r.memptr(), &chol_c);

        arma::uvec idx = find(U > 0);
        double perc = 100 * idx.n_elem / U.n_elem;
        stdout_printf("\t%d- res_norm = %f, U_density = %.2f%% (%d nnz)\n", i, a, perc, idx.n_elem);
        FLUSH;

        normM = normM - (r % r);
    }

    res.selected_cols = K;
    res.column_norms = norm_trace;

    cholmod_free_sparse(&AS, &chol_c);
    cholmod_free_sparse(&AS_sq, &chol_c);
    cholmod_finish(&chol_c);

    return res;
}
