#ifndef ACTIONET_UTILS_ACTIVE_SET_HPP
#define ACTIONET_UTILS_ACTIVE_SET_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_action_numeric_policy.hpp"

namespace actionet {

/// @brief Active-set method with direct inversion and update.
///
/// @param M Constraint matrix.
/// @param b Target vector.
/// @param lambda2 Regularization strength.
/// @param epsilon Convergence tolerance.
/// @return Solution vector.
arma::vec activeSet_arma(
    const arma::mat &M,
    const arma::vec &b,
    double lambda2 = utils_internal::action_numeric_policy::simplex_l2_regularization,
    double epsilon = utils_internal::action_numeric_policy::simplex_optimality_tolerance);

/// @brief Active-set method with cached Gram matrix.
///
/// @param M Constraint matrix.
/// @param b Target vector.
/// @param G Cached Gram matrix.
/// @param lambda2 Regularization strength.
/// @param epsilon Convergence tolerance.
/// @return Solution vector.
arma::vec activeSetS_arma(
    const arma::mat &M,
    const arma::vec &b,
    const arma::mat &G,
    double lambda2 = utils_internal::action_numeric_policy::simplex_l2_regularization,
    double epsilon = utils_internal::action_numeric_policy::simplex_optimality_tolerance);

} // namespace actionet

#endif //ACTIONET_UTILS_ACTIVE_SET_HPP
