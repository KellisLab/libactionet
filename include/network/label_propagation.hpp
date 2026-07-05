// Label propagation algorithm (LPA)
#ifndef ACTIONET_LABEL_PROPAGATION_HPP
#define ACTIONET_LABEL_PROPAGATION_HPP

#include "libactionet_config.hpp"

namespace actionet {
    /// @brief Run label propagation on a graph.
    ///
    /// @param G Graph adjacency matrix.
    /// @param labels Initial labels (numeric; -1 for unknown if used).
    /// @param lambda Propagation strength.
    /// @param iters Number of iterations.
    /// @param sig_threshold Significance threshold.
    /// @param fixed_labels 0-indexed positions of vertices whose labels are reverted after each iteration.
    /// @param thread_no Number of threads (0 = auto).
    ///
    /// @return Updated label vector.
    arma::vec runLPA(const arma::sp_mat& G, const arma::vec& labels, double lambda = 0, int iters = 3, double sig_threshold = 3,
                     arma::uvec fixed_labels = arma::uvec(), int thread_no = 0);
}

#endif //ACTIONET_LABEL_PROPAGATION_HPP
