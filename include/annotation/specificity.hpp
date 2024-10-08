#ifndef ACTIONET_SPECIFICITY_HPP
#define ACTIONET_SPECIFICITY_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(T& S, arma::mat& H, int thread_no = 0);

    template <typename T>
    arma::field<arma::mat> computeFeatureSpecificity(T& S, arma::uvec& labels, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_SPECIFICITY_HPP
