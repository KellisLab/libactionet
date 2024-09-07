#ifndef ACTIONET_MATRIX_MISC_HPP
#define ACTIONET_MATRIX_MISC_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    arma::mat compute_grouped_rowsums(arma::sp_mat& S, arma::vec& sample_assignments);

    arma::mat compute_grouped_rowsums(arma::mat& S, arma::vec& sample_assignments);

    template <typename T>
    arma::mat compute_grouped_rowmeans(T& S, arma::vec& sample_assignments);

    arma::mat compute_grouped_rowvars(arma::sp_mat& S, arma::vec sample_assignments);

    arma::mat compute_grouped_rowvars(arma::mat& S, arma::vec& sample_assignments);
} // namespace actionet

#endif //ACTIONET_MATRIX_MISC_HPP
