#include "network_diffusion_ext.h"

arma::mat PR_linsys(arma::sp_mat &G, arma::sp_mat &X, double alpha, int thread_no) {
    X = arma::normalise(X, 1, 0);

    arma::sp_mat P = arma::normalise(G, 1, 0);
    arma::sp_mat I = arma::speye(P.n_rows, P.n_cols);
    arma::sp_mat A = I - alpha * P;
    // arma::mat PR = (1-alpha)*arma::spsolve(A, arma::mat(X), "superlu");
    arma::mat PR = (1 - alpha) * arma::spsolve(A, arma::mat(X));

    return (PR);
}

arma::mat compute_network_diffusion_direct(arma::sp_mat &G, arma::sp_mat &X0, int thread_no, double alpha) {
    // A common struct that cholmod always needs
    cholmod_common c;
    cholmod_start(&c);

    int *Ti, *Tj;
    double *Tx;

    // Construct A
    // Construct transition matrix
    arma::vec d = arma::vec(arma::trans(arma::sum(G, 0)));
    arma::uvec zero_idx = arma::find(d == 0);
    d(zero_idx).ones();
    arma::sp_mat P = G;

    for (arma::sp_mat::iterator it = P.begin(); it != P.end(); ++it) {
        (*it) = (*it) / (std::sqrt(d(it.row()) * d(it.col())));
    }
    P = -alpha * P;
    P.diag().ones();

    stdout_printf("Creating A\n");
    FLUSH;
    cholmod_triplet *T = cholmod_allocate_triplet(P.n_rows, P.n_cols, P.n_nonzero,
                                                  0, CHOLMOD_REAL, &c);
    T->nnz = P.n_nonzero;
    Ti = static_cast<int *>(T->i);
    Tj = static_cast<int *>(T->j);
    Tx = static_cast<double *>(T->x);
    int idx = 0;
    for (arma::sp_mat::const_iterator it = P.begin(); it != P.end(); ++it) {
        Ti[idx] = it.row();
        Tj[idx] = it.col();
        Tx[idx] = (*it);
        idx++;
    }
    cholmod_sparse *A = cholmod_triplet_to_sparse(T, P.n_nonzero, &c);
    cholmod_free_triplet(&T, &c);

    // Construct B
    stdout_printf("Creating B\n");
    FLUSH;
    arma::vec d_X = arma::vec(arma::trans(arma::sum(X0, 0)));
    zero_idx = arma::find(d_X == 0);
    d_X(zero_idx).ones();
    arma::sp_mat D_X(X0.n_cols, X0.n_cols);
    D_X.diag() = d_X;

    X0 = arma::normalise(X0, 1, 0);

    T = cholmod_allocate_triplet(X0.n_rows, X0.n_cols, X0.n_nonzero, 0,
                                 CHOLMOD_REAL, &c);
    T->nnz = X0.n_nonzero;
    Ti = static_cast<int *>(T->i);
    Tj = static_cast<int *>(T->j);
    Tx = static_cast<double *>(T->x);
    idx = 0;
    for (arma::sp_mat::const_iterator it = X0.begin(); it != X0.end(); ++it) {
        Ti[idx] = it.row();
        Tj[idx] = it.col();
        Tx[idx] = (*it);
        idx++;
    }
    cholmod_sparse *B = cholmod_triplet_to_sparse(T, X0.n_nonzero, &c);
    cholmod_free_triplet(&T, &c);

    // Solve linear system
    stdout_printf("Chlmod analyze\n");
    FLUSH;
    cholmod_factor *L = cholmod_analyze(A, &c);
    stdout_printf("Chlmod factor\n");
    FLUSH;
    cholmod_factorize(A, L, &c);
    stdout_printf("Solve\n");
    FLUSH;
    cholmod_sparse *A_inv_B = cholmod_spsolve(CHOLMOD_A, L, B, &c);

    // Export results
    stdout_printf("Export\n");
    FLUSH;
    T = cholmod_sparse_to_triplet(A_inv_B, &c);
    Ti = (int *) T->i;
    Tj = (int *) T->j;
    Tx = (double *) T->x;
    arma::umat locations(2, T->nnz);
    for (int k = 0; k < T->nnz; k++) {
        locations(0, k) = Ti[k];
        locations(1, k) = Tj[k];
    }
    arma::mat PR = arma::mat(
            arma::sp_mat(locations, (1 - alpha) * arma::vec(Tx, T->nnz), X0.n_rows, X0.n_cols));

    PR = arma::normalise(PR, 1, 0);
    PR = PR * D_X;

    // Free up matrices
    cholmod_free_factor(&L, &c);
    cholmod_free_sparse(&A, &c);
    cholmod_free_sparse(&B, &c);
    cholmod_free_triplet(&T, &c);
    cholmod_finish(&c);

    return (PR);
}
