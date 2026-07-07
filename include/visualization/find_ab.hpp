// From umappp https://github.com/LTLA/umappp/blob/master/include/umappp/find_ab.hpp
// Vendored from upstream commit: (see src/visualization/find_ab.cpp for pinned SHA)
#ifndef UMAPPP_FIND_AB_HPP
#define UMAPPP_FIND_AB_HPP

#include <utility>

/*
 * This function attempts to find 'a' and 'b' to fit:
 *
 *     y ~ 1 / (1 + a * x^(2 * b))
 *
 * against the curve:
 *
 *     pmin(1, exp(-(x - d) / s))
 *
 * where 'd' is min_dist and 's' is spread.  Solved by Gauss-Newton with
 * Levenberg-Marquardt dampening on a fixed grid — see the .cpp for the full
 * derivation and comment history from the upstream umappp source.
 *
 * The template body previously lived in this header; it has been moved to
 * src/visualization/find_ab.cpp with an explicit instantiation for `float`,
 * which is the only in-tree specialisation used (via
 * `UwotArgs::set_ab`).  Keeping the declaration here preserves the include
 * contract for external consumers.
 */

template <typename Float_>
std::pair<Float_, Float_> find_ab(Float_ spread, Float_ min_dist);

// Explicit instantiations exist in src/visualization/find_ab.cpp for `float`.
// Adding new specialisations requires a matching instantiation there.

#endif // UMAPPP_FIND_AB_HPP
