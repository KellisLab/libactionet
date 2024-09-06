#ifndef ACTIONET_SPECIFICITY_HPP
#define ACTIONET_SPECIFICITY_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {

    arma::field<arma::mat> compute_feature_specificity(arma::sp_mat &S, arma::mat &H, int thread_no = 0);

    arma::field<arma::mat> compute_feature_specificity(arma::mat &S, arma::mat &H, int thread_no = 0);

    template <typename T>
    arma::field<arma::mat> compute_feature_specificity(T &S, arma::uvec sample_assignments, int thread_no = 0);

} // namespace actionet

#endif //ACTIONET_SPECIFICITY_HPP
