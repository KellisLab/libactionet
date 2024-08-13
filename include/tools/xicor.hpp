// Computes Xi correlation coefficient for vectors and matrices
// S.Chatterjee, A new coefficient of correlation (2019) (https://doi.org/10.48550/arXiv.1909.10140)
#ifndef LIBACTIONET_XICOR_HPP
#define LIBACTIONET_XICOR_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_stats.hpp"
#include "aarand/aarand.hpp"

namespace ACTIONet {

    arma::vec xicor(arma::vec xvec, arma::vec yvec, bool compute_pval = true, int seed = 0);

    arma::field<arma::mat> XICOR(arma::mat &X, arma::mat &Y, bool compute_pval = true, int seed = 0, int thread_no = 0);

}

#endif //LIBACTIONET_XICOR_HPP
