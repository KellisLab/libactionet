#include "utils_internal/utils_hnsw.hpp"
#include "hnsw/space_js.h"

// Obtain approximation algorithm
hnswlib::HierarchicalNSW<float> *
getApproximationAlgo(std::string distance_metric, arma::mat H, double M, double ef_construction) {
    int max_elements = H.n_cols;
    int dim = H.n_rows;
    // space to use determined by distance metric
    hnswlib::SpaceInterface<float> *space;
    if (distance_metric == "jsd") {
        space = new hnswlib::JSDSpace(dim); // JSD
    } else if (distance_metric == "l2") // l2
    {
        space = new hnswlib::L2Space(dim);
    } else {
        space = new hnswlib::InnerProductSpace(dim); // innerproduct
    }
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, max_elements, M,
                                                                                    ef_construction);
    return (appr_alg);
}
