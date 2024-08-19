// Main interface for reduction/decomposition algorithms
#ifndef ACTIONET_REDUCE_KERNEL_HPP
#define ACTIONET_REDUCE_KERNEL_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Entry point to compute a reduced kernel matrix
    template<typename T>
    arma::field<arma::mat> reduce_kernel(T &S, int dim, int iter, int seed, int SVD_algorithm, bool prenormalize,
                                         int verbose);
}

#endif //ACTIONET_REDUCE_KERNEL_HPP
