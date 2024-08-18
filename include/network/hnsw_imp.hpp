// Header implementation of necessary hnsw structures and functions for `build_network`
#ifndef ACTIONET_HNSW_IMP_HPP
#define ACTIONET_HNSW_IMP_HPP

#include "libactionet_config.hpp"
#include "hnswlib/hnswlib.h"
#include "hnsw_jensen_shannon.hpp"

// Structs: private
struct AddWorker {
    hnswlib::HierarchicalNSW<float> *hnsw;
    const arma::fmat &data;

    AddWorker(hnswlib::HierarchicalNSW<float> *hnsw,
              const arma::fmat &data) : hnsw(hnsw), data(data) {
    }

    void operator()(size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            hnsw->addPoint(data.colptr(i), i);
        }
    }
};

// Exceptions
// Throw if invalid distance metric has been specified
class invalidDistanceMetric : public std::exception {
    virtual const char *what() const throw() {
        return "Invalid distance metric specified for hnswlib; must be one of jsd, l2, ip";
    }
} distMetException;

// Throw if invalid nn approach has been specified
class invalidNNApproach : public std::exception {
    virtual const char *what() const throw() {
        return "Invalid nearest neighbors approach specified;  must be one of k*nn or knn";
    }
} nnApproachException;

// Functions: Must be header only. hnsw is allergic to implementation. Will break linking.
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

#endif //ACTIONET_HNSW_IMP_HPP
