#include "reduction_ext.h"

arma::field<arma::mat> SVD2ACTIONred(arma::sp_mat &S, arma::field<arma::mat> SVD_results) {
    stdout_printf("SVD => ACTIONred (sparse)\n");
    FLUSH;
    int n = S.n_rows;
    int dim = SVD_results(0).n_cols;

    // Update 1: Orthogonalize columns w.r.t. background (mean)
    arma::vec mu = arma::vec(arma::mean(S, 1));
    arma::vec v = mu / arma::norm(mu, 2);
    arma::vec a1 = v;
    arma::vec b1 = -arma::trans(S) * v;

    // Update 2: Center columns of orthogonalized matrix before performing SVD
    arma::vec c = arma::vec(arma::trans(arma::mean(S, 0)));
    double a1_mean = arma::mean(a1);
    arma::vec a2 = arma::ones(S.n_rows);
    arma::vec b2 = -(a1_mean * b1 + c);

    arma::mat A = arma::join_rows(a1, a2);
    arma::mat B = arma::join_rows(b1, b2);

    arma::field<arma::mat> perturbed_SVD = actionet::perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
}

arma::field<arma::mat> SVD2ACTIONred(arma::mat &S, arma::field<arma::mat> SVD_results) {
    stdout_printf("SVD => ACTIONred (dense)\n");
    FLUSH;
    int n = S.n_rows;
    int dim = SVD_results(0).n_cols;

    // Update 1: Orthogonalize columns w.r.t. background (mean)
    arma::vec mu = arma::vec(arma::mean(S, 1));
    arma::vec v = mu / arma::norm(mu, 2);
    arma::vec a1 = v;
    arma::vec b1 = -arma::trans(S) * v;

    // Update 2: Center columns of orthogonalized matrix before performing SVD
    arma::vec c = arma::vec(arma::trans(arma::mean(S, 0)));
    double a1_mean = arma::mean(a1);
    arma::vec a2 = arma::ones(S.n_rows);
    arma::vec b2 = -(a1_mean * b1 + c);

    arma::mat A = arma::join_rows(a1, a2);
    arma::mat B = arma::join_rows(b1, b2);

    arma::field<arma::mat> perturbed_SVD = actionet::perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
}

arma::field<arma::mat> PCA2ACTIONred(arma::sp_mat &S, arma::field<arma::mat> PCA_results) {
    stdout_printf("Reverting column-centering ... ");
    arma::field<arma::mat> SVD_results = actionet::PCA2SVD(S, PCA_results);
    stdout_printf("done\n");
    FLUSH;

    arma::field<arma::mat> output = SVD2ACTIONred(S, SVD_results);
    return output;
}

arma::field<arma::mat> PCA2ACTIONred(arma::mat &S, arma::field<arma::mat> PCA_results) {
    stdout_printf("Reverting column-centering ... ");
    arma::field<arma::mat> SVD_results = actionet::PCA2SVD(S, PCA_results);
    stdout_printf("done\n");
    FLUSH;

    arma::field<arma::mat> output = SVD2ACTIONred(S, SVD_results);
    return output;
}

arma::field<arma::mat> ACTIONred2SVD(arma::field<arma::mat> SVD_results) {
    stdout_printf("ACTION kernel => SVD\n");
    FLUSH;

    arma::mat A = -1 * SVD_results(3); // Reverting
    arma::mat B = SVD_results(4);

    arma::field<arma::mat> perturbed_SVD = actionet::perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
}
