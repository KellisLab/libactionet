#ifndef ACTIONET_MATRIX_MISC_HPP
#define ACTIONET_MATRIX_MISC_HPP

#include "libactionet_config.hpp"

// Exported
namespace ACTIONet {

    arma::mat compute_grouped_rowsums(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments);

    arma::mat compute_grouped_rowsums(arma::mat &S, arma::Col<unsigned long long> sample_assignments);

    arma::mat compute_grouped_rowmeans(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments);

    arma::mat compute_grouped_rowmeans(arma::mat &S, arma::Col<unsigned long long> sample_assignments);

    arma::mat compute_grouped_rowvars(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments);

    arma::mat compute_grouped_rowvars(arma::mat &S, arma::Col<unsigned long long> sample_assignments);

} // namespace ACTIONet

#endif //ACTIONET_MATRIX_MISC_HPP
