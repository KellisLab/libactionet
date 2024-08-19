// Main interface for reduction/decomposition algorithms
#ifndef ACTIONET_REDUCTION_HPP
#define ACTIONET_REDUCTION_HPP

#include "libactionet_config.hpp"

// Functions: internal
arma::field<arma::mat> deflate_reduction(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B);

// Exported
namespace actionet {
    // Entry point to compute a reduced kernel matrix
    template<typename T>
    arma::field<arma::mat> reduce_kernel(T &S, int dim, int iter, int seed, int SVD_algorithm, bool prenormalize,
                                         int verbose);

    // Convert between PCA and SVD results
    arma::field<arma::mat> PCA2SVD(arma::sp_mat &S, arma::field<arma::mat> PCA_results);

    arma::field<arma::mat> PCA2SVD(arma::mat &S, arma::field<arma::mat> PCA_results);

    arma::field<arma::mat> SVD2PCA(arma::sp_mat &S, arma::field<arma::mat> SVD_results);

    arma::field<arma::mat> SVD2PCA(arma::mat &S, arma::field<arma::mat> SVD_results);
}

#endif //ACTIONET_REDUCTION_HPP
