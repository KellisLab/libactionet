#ifndef ACTIONET_NORMALIZATION_EXT_H
#define ACTIONET_NORMALIZATION_EXT_H

#include "libactionet_config.hpp"

arma::mat renormalize_input_matrix(arma::mat &S, arma::Col<unsigned long long> sample_assignments);

arma::sp_mat renormalize_input_matrix(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments);

// TODO: Probably incorrect implementation
arma::sp_mat LSI(const arma::sp_mat& S, double size_factor = 100000);

#endif //ACTIONET_NORMALIZATION_EXT_H
