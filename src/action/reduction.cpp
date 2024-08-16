#include "action/reduction.hpp"
#include "action/svd.hpp"

arma::field<arma::mat> deflate_reduction(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B) {

    stdout_printf("\tDeflating reduction ... ");
    FLUSH;

    arma::vec mu_A = arma::vec(arma::trans(arma::mean(A, 0)));
    arma::vec mu = B * mu_A;

    A = arma::join_rows(arma::ones(A.n_rows), A);
    B = arma::join_rows(-mu, B);
    stdout_printf("done\n");
    FLUSH;

    arma::field<arma::mat> perturbed_SVD = ACTIONet::perturbedSVD(SVD_results, A, B);
    return (perturbed_SVD);
}

namespace ACTIONet {

    arma::field<arma::mat> reduce_kernel(arma::sp_mat &S, int dim, int iter, int seed, int SVD_algorithm,
                                         bool prenormalize, int verbose) {
        int n = S.n_rows;

        if (prenormalize)
            S = arma::normalise(S, 2);

        stdout_printf("Computing reduced ACTION kernel (sparse):\n");
        FLUSH;

        stdout_printf("\tPerforming SVD on original matrix: ");
        FLUSH;
        arma::vec s;
        arma::mat U, V;
        arma::field<arma::mat> SVD_results(3);

        switch (SVD_algorithm) {
            case FULL_SVD:
                arma::svd_econ(U, s, V, arma::mat(S));
                SVD_results(0) = U;
                SVD_results(1) = s;
                SVD_results(2) = V;
                break;
            case IRLB_ALG:
                SVD_results = IRLB_SVD(S, dim, iter, seed, verbose);
                break;
            case HALKO_ALG:
                SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
                break;
            case FENG_ALG:
                SVD_results = FengSVD(S, dim, iter, seed, verbose);
                break;
            default:
                stderr_printf("Unknown SVD algorithm chosen (%d). Switching to Halko.\n", SVD_algorithm);
                FLUSH;
                SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
                break;
        }

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

        arma::field<arma::mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

        return perturbed_SVD;
    }

    arma::field<arma::mat> reduce_kernel(arma::mat &S, int dim, int iter, int seed, int SVD_algorithm,
                                         bool prenormalize, int verbose) {
        int n = S.n_rows;

        if (prenormalize)
            S = arma::normalise(S, 2);

        stdout_printf("Computing reduced ACTION kernel (dense):\n");
        FLUSH;
        stdout_printf("\tPerforming SVD on original matrix: ");
        FLUSH;

        arma::vec s;
        arma::mat U, V;
        arma::field<arma::mat> SVD_results(3);
        switch (SVD_algorithm) {
            case FULL_SVD:
                arma::svd_econ(U, s, V, S);
                SVD_results(0) = U;
                SVD_results(1) = s;
                SVD_results(2) = V;
                break;
            case IRLB_ALG:
                SVD_results = IRLB_SVD(S, dim, iter, seed, verbose);
                break;
            case HALKO_ALG:
                SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
                break;
            case FENG_ALG:
                SVD_results = FengSVD(S, dim, iter, seed, verbose);
                break;
            default:
                stderr_printf("Unknown SVD algorithm chosen (%d). Switching to Halko.\n", SVD_algorithm);
                FLUSH;
                SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
                break;
        }

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

        arma::field<arma::mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

        return perturbed_SVD;
    }

    arma::field<arma::mat> PCA2SVD(arma::sp_mat &S, arma::field<arma::mat> PCA_results) {
        int n = S.n_rows;

        stdout_printf("PCA => SVD (sparse)\n");
        FLUSH;
        arma::mat U = PCA_results(0);
        arma::vec s = PCA_results(1);
        arma::mat V = PCA_results(2);

        int dim = U.n_cols;

        arma::mat A = arma::ones(S.n_rows, 1);
        arma::mat B = arma::mat(arma::trans(arma::mean(S, 0)));

        arma::field<arma::mat> perturbed_SVD = perturbedSVD(PCA_results, A, B);

        return perturbed_SVD;
    }

    arma::field<arma::mat> PCA2SVD(arma::mat &S, arma::field<arma::mat> PCA_results) {
        int n = S.n_rows;

        stdout_printf("PCA => SVD (dense)\n");
        FLUSH;
        arma::mat U = PCA_results(0);
        arma::vec s = PCA_results(1);
        arma::mat V = PCA_results(2);

        int dim = U.n_cols;

        arma::mat A = arma::ones(S.n_rows, 1);
        arma::mat B = arma::mat(arma::trans(arma::mean(S, 0)));

        arma::field<arma::mat> perturbed_SVD = perturbedSVD(PCA_results, A, B);

        return perturbed_SVD;
    }

    arma::field<arma::mat> SVD2PCA(arma::sp_mat &S, arma::field<arma::mat> SVD_results) {
        int n = S.n_rows;

        stdout_printf("SVD => PCA (sparse)\n");
        FLUSH;
        arma::mat U = SVD_results(0);
        arma::vec s = SVD_results(1);
        arma::mat V = SVD_results(2);

        int dim = U.n_cols;

        arma::mat A = arma::ones(S.n_rows, 1);
        arma::mat B = -arma::mat(arma::trans(arma::mean(S, 0)));

        arma::field<arma::mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

        return perturbed_SVD;
    }

    arma::field<arma::mat> SVD2PCA(arma::mat &S, arma::field<arma::mat> SVD_results) {
        int n = S.n_rows;

        stdout_printf("SVD => PCA (dense)\n");
        FLUSH;
        arma::mat U = SVD_results(0);
        arma::vec s = SVD_results(1);
        arma::mat V = SVD_results(2);

        int dim = U.n_cols;

        arma::mat A = arma::ones(S.n_rows, 1);
        arma::mat B = -arma::mat(arma::trans(arma::mean(S, 0)));

        arma::field<arma::mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

        return perturbed_SVD;
    }

} // namespace ACTIONet
