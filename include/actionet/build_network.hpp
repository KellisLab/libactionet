#ifndef BUILD_NETWORK_HPP
#define BUILD_NETWORK_HPP

#include "actionet.hpp"
#include "utils/utils_parallel.hpp"
#include "hnsw/hnswlib.h"
#include "hnsw/space_js.h"
#include <set>

struct AddWorker
{
    hnswlib::HierarchicalNSW<float> *hnsw;
    const arma::fmat &data;

    AddWorker(hnswlib::HierarchicalNSW<float> *hnsw,
              const arma::fmat &data) : hnsw(hnsw), data(data)
    {
    }

    void operator()(size_t begin, size_t end)
    {
        for (size_t i = begin; i < end; i++)
        {
            hnsw->addPoint(data.colptr(i), i);
        }
    }
};

// Option arguments
// allowed distance metrics for hnswlib
std::set<std::string> distance_metrics = {"jsd", "l2", "ip"};

// allowed nn approaches
std::set<std::string> nn_approaches = {"k*nn", "knn"};

// Exceptions
// Throw if invalid distance metric has been specified
class invalidDistanceMetric : public std::exception
{
    virtual const char *what() const throw()
    {
        return "Invalid distance metric specified for hnswlib; must be one of jsd, l2, ip";
    }
} distMetException;

// Throw if invalid nn approach has been specified
class invalidNNApproach : public std::exception
{
    virtual const char *what() const throw()
    {
        return "Invalid nearest neighbors approach specified;  must be one of k*nn or knn";
    }
} nnApproachException;

// Functions
hnswlib::HierarchicalNSW<float> *getApproximationAlgo(std::string distance_metric, arma::mat H, double M, double ef_construction);

arma::sp_mat buildNetwork_KstarNN(arma::mat H, double density = 1.0, int thread_no = 0, double M = 16, double ef_construction = 200,
                                  double ef = 200, bool mutual_edges_only = true, std::string distance_metric = "jsd");

arma::sp_mat buildNetwork_KNN(arma::mat H, int k, int thread_no = 0, double M = 16, double ef_construction = 200, double ef = 200,
                              bool mutual_edges_only = true, std::string distance_metric = "jsd");

#endif
