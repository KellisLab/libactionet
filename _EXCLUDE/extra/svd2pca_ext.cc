#include "svd2pca_ext.h"
#include "decomposition/svd_main.hpp"

// FIXME: One of these is obviously wrong. They're identical.
// If kept, needs explicit instantiation.
template <typename T>
arma::field<arma::mat> SVD2PCA(T &S, arma::field<arma::mat> SVD_results) {
    stdout_printf("SVD => PCA\n");
    FLUSH;
    arma::mat U = SVD_results(0);
    arma::vec s = SVD_results(1);
    arma::mat V = SVD_results(2);

    arma::mat A = arma::ones(S.n_rows, 1);
    arma::mat B = -arma::mat(arma::trans(arma::mean(S, 0)));

    arma::field<arma::mat> perturbed_SVD = actionet::perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
}

// Probably this one needs to go.
template <typename T>
arma::field<arma::mat> PCA2SVD(T &S, arma::field<arma::mat> PCA_results) {
    stdout_printf("PCA => SVD\n");
    FLUSH;
    arma::mat U = PCA_results(0);
    arma::vec s = PCA_results(1);
    arma::mat V = PCA_results(2);

    arma::mat A = arma::ones(S.n_rows, 1);
    arma::mat B = arma::mat(arma::trans(arma::mean(S, 0)));

    arma::field<arma::mat> perturbed_SVD = actionet::perturbedSVD(PCA_results, A, B);

    return perturbed_SVD;
}
