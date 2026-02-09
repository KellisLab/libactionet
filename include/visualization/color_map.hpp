#ifndef ACTIONET_COLOR_MAP_HPP
#define ACTIONET_COLOR_MAP_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Compute RGB colors for nodes from 3D coordinates.
    ///
    /// @param coordinates Embedding coordinates (cells x 3).
    /// @param thread_no Number of threads (0 = auto).
    /// @return RGB colors (cells x 3, range [0,1]).
    arma::mat computeNodeColors(const arma::mat& coordinates, int thread_no);
} // actionet

#endif //ACTIONET_COLOR_MAP_HPP
