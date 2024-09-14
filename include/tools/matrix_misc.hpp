#ifndef ACTIONET_MATRIX_MISC_HPP
#define ACTIONET_MATRIX_MISC_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    arma::mat computeGroupedRowSums(arma::sp_mat& S, arma::vec& sample_assignments);

    arma::mat computeGroupedRowSums(arma::mat& S, arma::vec& sample_assignments);

    template <typename T>
    arma::mat computeGroupedRowMeans(T& S, arma::vec& sample_assignments);

    arma::mat computeGroupedRowVars(arma::sp_mat& S, arma::vec& sample_assignments);

    arma::mat computeGroupedRowVars(arma::mat& S, const arma::vec& sample_assignments);
} // namespace actionet

#endif //ACTIONET_MATRIX_MISC_HPP
