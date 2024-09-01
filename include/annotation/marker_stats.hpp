#ifndef ACTIONET_MARKER_STATS_HPP
#define ACTIONET_MARKER_STATS_HPP

#include "libactionet_config.hpp"

namespace actionet {
    // TODO: Likely obsolote. Replaced by aggregate_genesets_vision()
    arma::mat compute_marker_aggregate_stats(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& marker_mat,
                                             double alpha = 0.85, int max_it = 5, int thread_no = 0,
                                             bool ignore_baseline_expression = false);

    arma::field<arma::mat> aggregate_genesets_vision(arma::sp_mat& G, arma::sp_mat& S, arma::mat& X,
                                                     int network_normalization_method = 0, double alpha = 0.85,
                                                     int thread_no = 0);

}

#endif //ACTIONET_MARKER_STATS_HPP
