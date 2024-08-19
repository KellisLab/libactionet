#include "action/reduce_kernel.hpp"
#include "decomposition/svd_main.hpp"

namespace actionet {
    template<typename T>
    arma::field<arma::mat> reduce_kernel(T &S, int dim, int iter, int seed, int SVD_algorithm, bool prenormalize,
                                         int verbose) {
        if (prenormalize)
            S = arma::normalise(S, 2);

        if (verbose) {
            stdout_printf("Computing reduced ACTION kernel:\n");
            FLUSH;
        }

        arma::field<arma::mat> SVD_results(3);

        SVD_results = runSVD(S, dim, iter, seed, SVD_algorithm, verbose);

        // Update 1: Orthogonalize columns w.r.t. background (mean)
        arma::vec mu = arma::vec(arma::mean(S, 1));
        arma::vec a1 = mu / arma::norm(mu, 2);
        // arma::vec a1 = v;
        arma::vec b1 = -arma::trans(S) * a1;

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

    template arma::field<arma::mat> reduce_kernel<arma::mat>(arma::mat &S, int dim, int iter, int seed,
                                                             int SVD_algorithm, bool prenormalize, int verbose);

    template arma::field<arma::mat> reduce_kernel<arma::sp_mat>(arma::sp_mat &S, int dim, int iter, int seed,
                                                                int SVD_algorithm, bool prenormalize, int verbose);
} // namespace actionet
