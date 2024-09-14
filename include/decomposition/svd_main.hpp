// Interface for value decomposition (SVD) algorithms
#ifndef ACTIONET_SVD_MAIN_HPP
#define ACTIONET_SVD_MAIN_HPP

#include "libactionet_config.hpp"

// SVD algorithm options
#define ALG_IRLB 0
#define ALG_HALKO 1
#define ALG_FENG 2

// Exported
namespace actionet {
    template <typename T>
    arma::field<arma::mat> runSVD(T& A, int k, int max_it = 0, int seed = 0, int algorithm = ALG_IRLB, int verbose = 0);

    arma::field<arma::mat> perturbedSVD(arma::field<arma::mat>& SVD_results, arma::mat& A, arma::mat& B);
} // namespace actionet

#endif //ACTIONET_SVD_MAIN_HPP
