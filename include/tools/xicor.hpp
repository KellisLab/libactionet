// Computes Xi correlation coefficient for vectors and matrices
// S.Chatterjee, A new coefficient of correlation (2019) (https://doi.org/10.48550/arXiv.1909.10140)
#ifndef ACTIONET_XICOR_HPP
#define ACTIONET_XICOR_HPP

#include "libactionet_config.hpp"

namespace actionet {
    arma::vec xicor(arma::vec xvec, arma::vec yvec, bool compute_pval = true, int seed = 0);

    arma::field<arma::mat> XICOR(arma::mat& X, arma::mat& Y, bool compute_pval = true, int seed = 0, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_XICOR_HPP
