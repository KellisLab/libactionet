#ifndef ACTIONET_UTILS_HNSW_HPP
#define ACTIONET_UTILS_HNSW_HPP

#include "libactionet_config.hpp"
#include "hnsw/hnswlib.h"
#include "hnsw/space_js.h"

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

// Functions: private
hnswlib::HierarchicalNSW<float> *
getApproximationAlgo(std::string distance_metric, arma::mat H, double M, double ef_construction);

#endif //ACTIONET_UTILS_HNSW_HPP
