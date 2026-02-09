// Main interface for reduction/decomposition algorithms
#ifndef ACTIONET_REDUCE_KERNEL_HPP
#define ACTIONET_REDUCE_KERNEL_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Compute a reduced kernel matrix using truncated SVD.
    ///
    /// @tparam T Dense or sparse Armadillo matrix type.
    /// @param S Input matrix (<em>vars</em> x <em>obs</em>).
    /// @param k Number of singular vectors to estimate.
    /// @param svd_alg SVD algorithm (see <code>runSVD()</code>).
    /// @param max_it Maximum number of SVD iterations.
    /// @param seed Random seed.
    /// @param verbose Print status messages.
    ///
    /// @return Field with 5 elements: {S_r, sigma, U, A, B}.
    template <typename T>
    arma::field<arma::mat> reduceKernel(T& S, int k, int svd_alg = 0, int max_it = 0,
                                        int seed = 0, bool verbose = true);
}

#endif //ACTIONET_REDUCE_KERNEL_HPP
