// Orthogonalization method for batch correction
#ifndef ACTIONET_ORTHOGONALIZATION_HPP
#define ACTIONET_ORTHOGONALIZATION_HPP

#include "libactionet_config.hpp"

// Functions: internal
arma::field<arma::mat> deflate_reduction(arma::field<arma::mat> SVD_results, arma::mat& A, arma::mat& B);

namespace actionet {
    template <typename T>
    arma::field<arma::mat> orthogonalize_batch_effect(T& S, arma::field<arma::mat> SVD_results, arma::mat& design);

    template <typename T>
    arma::field<arma::mat> orthogonalize_basal(T& S, arma::field<arma::mat> SVD_results, arma::mat& basal);
} // namespace actionet

#endif //ACTIONET_ORTHOGONALIZATION_HPP
