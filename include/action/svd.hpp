// Singular value decomposition (SVD) algorithms
#ifndef ACTIONET_SVD_HPP
#define ACTIONET_SVD_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {

    arma::field<arma::mat> perturbedSVD(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B);

    arma::field<arma::mat> IRLB_SVD(arma::sp_mat &A, int dim, int iters = 1000, int seed = 0, int verbose = 1);

    arma::field<arma::mat> IRLB_SVD(arma::mat &A, int dim, int iters = 1000, int seed = 0, int verbose = 1);

    //******************************************************************************************************************
    // From: Xu Feng, Yuyang Xie, and Yaohang Li, "Fast Randomzied SVD for Sparse
    // Data," in Proc. the 10th Asian Conference on Machine Learning (ACML),
    // Beijing, China, Nov. 2018.
    //******************************************************************************************************************
    arma::field<arma::mat> FengSVD(arma::sp_mat &A, int dim, int iters, int seed = 0, int verbose = 1);

    arma::field<arma::mat> FengSVD(arma::mat &A, int dim, int iters, int seed = 0, int verbose = 1);

    //******************************************************************************************************************
    // From: N Halko, P. G Martinsson, and J. A Tropp. Finding structure with
    // randomness: Probabilistic algorithms for constructing approximate matrix
    // decompositions. Siam Review, 53(2):217-288, 2011.
    //******************************************************************************************************************
    arma::field<arma::mat> HalkoSVD(arma::sp_mat &A, int dim, int iters, int seed = 0, int verbose = 1);

    arma::field<arma::mat> HalkoSVD(arma::mat &A, int dim, int iters, int seed = 0, int verbose = 1);

} // namespace actionet

#endif //ACTIONET_SVD_HPP
