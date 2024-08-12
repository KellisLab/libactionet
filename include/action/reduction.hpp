// Main interface for reduction/decomposition algorithms
#ifndef LIBACTIONET_REDUCTION_HPP
#define LIBACTIONET_REDUCTION_HPP

#include "libactionet_config.hpp"
#include "action/svd.hpp"

// SVD algorithm options
#define FULL_SVD -1
#define IRLB_ALG 0
#define HALKO_ALG 1
#define FENG_ALG 2

// Functions: private
arma::field<arma::mat> deflate_reduction(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B);

// Exported
namespace ACTIONet {

    // Entry-points to compute a reduced kernel matrix
    arma::field<arma::mat> reduce_kernel(arma::sp_mat &S, int dim, int iter = 5, int seed = 0,
                                         int SVD_algorithm = HALKO_ALG, bool prenormalize = false, int verbose = 1);

    arma::field<arma::mat> reduce_kernel(arma::mat &S, int dim, int iter = 5, int seed = 0,
                                         int SVD_algorithm = HALKO_ALG, bool prenormalize = false, int verbose = 1);

    // Convert between PCA and SVD results
    arma::field<arma::mat> PCA2SVD(arma::sp_mat &S, arma::field<arma::mat> PCA_results);

    arma::field<arma::mat> PCA2SVD(arma::mat &S, arma::field<arma::mat> PCA_results);

    arma::field<arma::mat> SVD2PCA(arma::sp_mat &S, arma::field<arma::mat> SVD_results);

    arma::field<arma::mat> SVD2PCA(arma::mat &S, arma::field<arma::mat> SVD_results);
}

#endif //LIBACTIONET_REDUCTION_HPP
