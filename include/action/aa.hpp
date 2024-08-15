// Solves the standard Archetypal Analysis (AA) problem
#ifndef LIBACTIONET_AA_HPP
#define LIBACTIONET_AA_HPP

#include "libactionet_config.hpp"
#include "action/simplex_regression.hpp"
//#include <cblas.h>

// Exported
namespace ACTIONet {

    arma::field<arma::mat> run_AA(arma::mat &A, arma::mat &W0, int max_it = 100, double min_delta = 1e-6);

} // namespace ACTIONet

#endif //LIBACTIONET_AA_HPP
