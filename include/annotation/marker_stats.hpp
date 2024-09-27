#ifndef ACTIONET_MARKER_STATS_HPP
#define ACTIONET_MARKER_STATS_HPP

#include "libactionet_config.hpp"

namespace actionet {
    // TODO: Likely obsolete. Replaced by computeFeatureStatsVision()
    arma::mat computeFeatureStats(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X,
                                  int norm_type = 2, double alpha = 0.85, int max_it = 5,
                                  int thread_no = 0, bool ignore_baseline_expression = false);

    // norm_type: 0 (pagerank), 2 (sym_pagerank; recommended)
    arma::mat computeFeatureStatsVision(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X,
                                                     int norm_type = 2, double alpha = 0.85,
                                                     int max_it = 5, int thread_no = 0);
}

#endif //ACTIONET_MARKER_STATS_HPP
