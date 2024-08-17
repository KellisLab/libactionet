// Solves the standard Archetypal Analysis (AA) problem
#ifndef ACTIONET_AA_HPP
#define ACTIONET_AA_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {

    arma::field<arma::mat> run_AA(arma::mat &A, arma::mat &W0, int max_it = 100, double min_delta = 1e-6);

} // namespace actionet

#endif //ACTIONET_AA_HPP
