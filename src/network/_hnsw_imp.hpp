// Header implementation of necessary hnsw structures and functions for `build_network`
#ifndef ACTIONET_HNSW_IMP_HPP
#define ACTIONET_HNSW_IMP_HPP

#include "libactionet_config.hpp"
#include "hnswlib/hnswlib.h"
#include "_hnsw_jensen_shannon.hpp"
#include <type_traits>

static_assert(std::is_integral_v<hnswlib::labeltype>, "HNSW labels must remain integral");

// Functions: Must be header only. hnsw is allergic to implementation. Will break linking.

// RAII owner for a (SpaceInterface, HierarchicalNSW) pair.
// HierarchicalNSW stores a raw SpaceInterface* internally but does not delete it on
// destruction, so we track both and delete them together.
struct HnswIndex {
    hnswlib::SpaceInterface<float>*  space = nullptr;
    hnswlib::HierarchicalNSW<float>* hnsw  = nullptr;

    ~HnswIndex() { delete hnsw; delete space; }

    HnswIndex() = default;
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    HnswIndex(HnswIndex&& o) noexcept : space(o.space), hnsw(o.hnsw)
        { o.space = nullptr; o.hnsw = nullptr; }
};

// Allocate a SpaceInterface for the given metric and dimensionality.
namespace actionet {

inline hnswlib::SpaceInterface<float>*
makeHnswSpace(const std::string& distance_metric, int dim) {
    if (distance_metric == "jsd")
        return new hnswlib::JSDSpace(dim);
    if (distance_metric == "l2")
        return new hnswlib::L2Space(dim);
    return new hnswlib::InnerProductSpace(dim);
}

// Build an HNSW index from raw dimensions.  Returns an HnswIndex that owns
// both the space and the HierarchicalNSW objects.
inline HnswIndex
makeHnswIndex(const std::string& distance_metric,
              std::size_t        max_elements,
              int                dim,
              double             M,
              double             ef_construction) {
    HnswIndex idx;
    idx.space = makeHnswSpace(distance_metric, dim);
    idx.hnsw  = new hnswlib::HierarchicalNSW<float>(idx.space, max_elements, M, ef_construction);
    return idx;
}

} // namespace actionet


#endif //ACTIONET_HNSW_IMP_HPP
