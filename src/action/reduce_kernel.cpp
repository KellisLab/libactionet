#include "action/reduce_kernel.hpp"
#include "decomposition/svd_main.hpp"

namespace actionet {
    template <typename T>
    arma::field<arma::mat> reduceKernel(T& S, int dim, int svd_alg, int max_it, int seed, int verbose) {
        if (verbose) {
            stdout_printf("Computing reduced ACTION kernel:\n");
            FLUSH;
        }

        arma::field<arma::mat> SVD_results(3);

        SVD_results = runSVD(S, dim, max_it, seed, svd_alg, verbose);

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

        arma::field<arma::mat> reduction = perturbedSVD(SVD_results, A, B);

        // reduced kernel output
        arma::field<arma::mat> out(5); // out: V, sigma, S_r, A, B

        arma::vec sigma = reduction(1).col(0);

        double epsilon = 0.01 / std::sqrt(reduction(2).n_rows);
        arma::mat V = arma::round(reduction(2) / epsilon) * epsilon;

        for (int i = 0; i < V.n_cols; i++) {
            arma::vec v = V.col(i) * sigma(i);
            V.col(i) = v;
        }
        V = arma::trans(V);

        out(0) = V.eval(); // S_r (trans() is delayed evaluation)
        out(1) = sigma; // sigma
        out(2) = reduction(0); // V
        out(3) = reduction(3); // A
        out(4) = reduction(4); // B

        return out;
    }

    template arma::field<arma::mat> reduceKernel<arma::mat>(arma::mat& S, int dim, int svd_alg,
                                                             int iter, int seed, int verbose);

    template arma::field<arma::mat> reduceKernel<arma::sp_mat>(arma::sp_mat& S, int dim, int svd_alg,
                                                                int iter, int seed, int verbose);
} // namespace actionet
