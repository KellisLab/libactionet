// Orthogonalization method for batch correction
#ifndef ACTIONET_ORTHOGONALIZATION_HPP
#define ACTIONET_ORTHOGONALIZATION_HPP

#include "libactionet_config.hpp"

namespace actionet {

    arma::field<arma::mat>
    orthogonalize_batch_effect(arma::sp_mat S, arma::field<arma::mat> SVD_results, arma::mat &design);

    arma::field<arma::mat>
    orthogonalize_batch_effect(arma::mat &S, arma::field<arma::mat> SVD_results, arma::mat &design);

    arma::field<arma::mat> orthogonalize_basal(arma::sp_mat &S, arma::field<arma::mat> SVD_results, arma::mat &basal);

    arma::field<arma::mat> orthogonalize_basal(arma::mat &S, arma::field<arma::mat> SVD_results, arma::mat &basal);

} // namespace actionet

#endif //ACTIONET_ORTHOGONALIZATION_HPP
