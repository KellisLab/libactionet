// Orthogonalization method for batch correction
#ifndef ACTIONET_ORTHOGONALIZATION_HPP
#define ACTIONET_ORTHOGONALIZATION_HPP

#include "libactionet_config.hpp"

// Functions: internal
arma::field<arma::mat> deflateReduction(arma::field<arma::mat>& SVD_results, arma::mat& A, arma::mat& B);

namespace actionet {
    template <typename T>
    arma::field<arma::mat> orthogonalizeBatchEffect(T& S, arma::field<arma::mat>& SVD_results, arma::mat& design);

    template <typename T>
    arma::field<arma::mat> orthogonalizeBasal(T& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);
} // namespace actionet

#endif //ACTIONET_ORTHOGONALIZATION_HPP
