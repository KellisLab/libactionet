#ifndef ACTIONET_ENRICHMENT_HPP
#define ACTIONET_ENRICHMENT_HPP

#include "libactionet_config.hpp"

namespace ACTIONet {

    arma::mat assess_label_enrichment(arma::sp_mat &H, arma::mat &M, int thread_no = 1);

    arma::field<arma::mat> assess_enrichment(arma::mat &scores, arma::sp_mat &associations, int thread_no = 1);
    
} // namespace ACTIONet

#endif //ACTIONET_ENRICHMENT_HPP
