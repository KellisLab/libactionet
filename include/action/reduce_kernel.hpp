// Main interface for reduction/decomposition algorithms
#ifndef ACTIONET_REDUCE_KERNEL_HPP
#define ACTIONET_REDUCE_KERNEL_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Entry point to compute a reduced kernel matrix
    /// @brief Compute reduced kernel matrix
    ///
    /// @param S Input matrix (<em>vars</em> x <em>obs</em>).
    /// May be <code>arma::mat</code> or <code>arma::sp_mat</code>.
    /// @param k Number of singular vectors to estimate. Passed to <code>runSVD()</code>.
    /// @param svd_alg Singular value decomposition algorithm. See to <code>runSVD()</code> for options.
    /// @param max_it Maximum number of iterations. Passed to <code>runSVD()</code>.
    /// @param seed Random seed.
    /// @param verbose Print status messages.
    ///
    /// @return Field with 5 elements:
    /// - 0: <code>arma::mat</code> Reduced kernel matrix.
    /// - 1: <code>arma::vec</code> Singular values.
    /// - 2: <code>arma::mat</code> Left singular vectors.
    /// - 3: <code>arma::mat</code> <b>A</b> perturbation matrix.
    /// - 4: <code>arma::mat</code> <b>B</b> perturbation matrix.
    ///
    /// @remark See <code>runSVD()</code>.
    template <typename T>
    arma::field<arma::mat> reduceKernel(T& S, int k, int svd_alg = 0, int max_it = 0,
                                        int seed = 0, int verbose = 1);
}

#endif //ACTIONET_REDUCE_KERNEL_HPP
