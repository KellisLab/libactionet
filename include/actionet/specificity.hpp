#ifndef LIBACTIONET_SPECIFICITY_HPP
#define LIBACTIONET_SPECIFICITY_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_matrix.hpp"

// Exported
namespace ACTIONet {

    arma::field<arma::mat> compute_feature_specificity(arma::sp_mat &S, arma::mat &H, int thread_no = 0);

    arma::field<arma::mat> compute_feature_specificity(arma::mat &S, arma::mat &H, int thread_no = 0);

    arma::field<arma::mat>
    compute_feature_specificity(arma::sp_mat &S, arma::uvec sample_assignments, int thread_no = 0);

    arma::field<arma::mat> compute_feature_specificity(arma::mat &S, arma::uvec sample_assignments, int thread_no = 0);

} // namespace ACTIONet

#endif //LIBACTIONET_SPECIFICITY_HPP
