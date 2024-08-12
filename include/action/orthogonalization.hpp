// Orthogonalization method for batch correction
#ifndef LIBACTIONET_ORTHOGONALIZATION_HPP
#define LIBACTIONET_ORTHOGONALIZATION_HPP

#include "libactionet_config.hpp"

#include "action/svd.hpp"
#include "action/reduction.hpp"

namespace ACTIONet {

    arma::field<arma::mat>
    orthogonalize_batch_effect(arma::sp_mat &S, arma::field<arma::mat> SVD_results, arma::mat &design);

    arma::field<arma::mat>
    orthogonalize_batch_effect(arma::mat &S, arma::field<arma::mat> SVD_results, arma::mat &design);

    arma::field<arma::mat> orthogonalize_basal(arma::sp_mat &S, arma::field<arma::mat> SVD_results, arma::mat &basal);

    arma::field<arma::mat> orthogonalize_basal(arma::mat &S, arma::field<arma::mat> SVD_results, arma::mat &basal);

}

#endif //LIBACTIONET_ORTHOGONALIZATION_HPP
