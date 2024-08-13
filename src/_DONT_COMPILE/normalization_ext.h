#ifndef LIBACTIONET_NORMALIZATION_EXT_H
#define LIBACTIONET_NORMALIZATION_EXT_H

#include "libactionet_config.hpp"

arma::mat renormalize_input_matrix(arma::mat &S, arma::Col<unsigned long long> sample_assignments);

arma::sp_mat renormalize_input_matrix(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments);

#endif //LIBACTIONET_NORMALIZATION_EXT_H
