#ifndef ACTIONET_MATRIX_MISC_HPP
#define ACTIONET_MATRIX_MISC_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    arma::mat computeGroupedSums(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
    arma::sp_mat computeGroupedSums2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
    
    arma::mat computeGroupedSums(arma::mat& S, arma::vec& sample_assignments, int axis = 0);
    
    template <typename T>
    arma::mat computeGroupedMeans(T& S, arma::vec& sample_assignments, int axis = 0);
    arma::sp_mat computeGroupedMeans2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
    
    arma::mat computeGroupedVars(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
    arma::mat computeGroupedVars(arma::mat& S, arma::vec& sample_assignments, int axis = 0);
    arma::sp_mat computeGroupedVars2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0);
} // namespace actionet

#endif //ACTIONET_MATRIX_MISC_HPP
