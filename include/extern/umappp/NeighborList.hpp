#ifndef UMAPPP_NEIGHBOR_LIST_HPP
#define UMAPPP_NEIGHBOR_LIST_HPP

#include <utility>
#include <vector>

// #include "knncolle/knncolle.hpp"

/**
 * @file NeighborList.hpp
 *
 * @brief Defines the `NeighborList` alias.
 */

namespace umappp {

// /**
//  * @brief Lists of neighbors for each observation.
//  *
//  * @tparam Index_ Integer type of the neighbor indices.
//  * @tparam Float_ Floating-point type for the distances.
//  *
//  * This is a convenient alias for the `knncolle::NeighborList` class.
//  * Each inner vector corresponds to an observation and contains the list of nearest neighbors for that observation, sorted by increasing distance.
//  * Neighbors for each observation should be unique - there should be no more than one occurrence of each index in each inner vector.
//  * Also, the inner vector for observation `i` should not contain any `Neighbor` with index `i`.
//  */
// template<typename Index_, typename Float_>
// using NeighborList = knncolle::NeighborList<Index_, Float_>;

 /**
  * List of nearest neighbors for multiple observations.
  * Each entry corresponds to an observation and contains a nested list (i.e., vector) of its neighbors.
  * Each entry of the nested vector is a pair that contains the identity of the neighbor as an observation index (first)
  * and the distance from the observation to the neighbor (second), sorted by increasing distance.
  *
  * @tparam Index_ Integer type for the indices.
  * @tparam Float_ Floating point type for the distances.
  */
 template<typename Index_ = int, typename Float_ = double>
 using NeighborList = std::vector<std::vector<std::pair<Index_, Float_> > >;

}

#endif
