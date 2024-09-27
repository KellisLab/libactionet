#ifndef ACTIONET_ENRICHMENT_HPP
#define ACTIONET_ENRICHMENT_HPP

#include "libactionet_config.hpp"

namespace actionet {
    arma::mat computeGraphLabelEnrichment(const arma::sp_mat& G, arma::mat& scores, int thread_no = 1);

    arma::field<arma::mat> assess_enrichment(arma::mat& scores, arma::sp_mat& associations, int thread_no = 1);
} // namespace actionet

#endif //ACTIONET_ENRICHMENT_HPP
