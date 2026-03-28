#ifndef ACTIONET_MATRIX_MISC_HPP
#define ACTIONET_MATRIX_MISC_HPP

#include "libactionet_config.hpp"

namespace actionet {

    /// @brief Compute grouped sums along an axis.
    template <typename InputT, typename OutputT = arma::mat>
    OutputT computeGroupedSums(const InputT& S, const arma::vec& sample_assignments, int axis = 0);

    /// @brief Compute grouped means along an axis.
    template <typename InputT, typename OutputT = arma::mat>
    OutputT computeGroupedMeans(const InputT& S, const arma::vec& sample_assignments, int axis = 0);

    /// @brief Compute grouped variances along an axis.
    template <typename InputT, typename OutputT = arma::mat>
    OutputT computeGroupedVars(const InputT& S, const arma::vec& sample_assignments, int axis = 0);

} // namespace actionet

#endif //ACTIONET_MATRIX_MISC_HPP
