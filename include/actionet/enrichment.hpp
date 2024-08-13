#ifndef LIBACTIONET_ENRICHMENT_HPP
#define LIBACTIONET_ENRICHMENT_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_matrix.hpp"

namespace ACTIONet {

    arma::mat assess_label_enrichment(arma::sp_mat &H, arma::mat &M, int thread_no = 1);

    arma::field<arma::mat> assess_enrichment(arma::mat &scores, arma::sp_mat &associations, int thread_no = 1);
    
} // namespace ACTIONet

#endif //LIBACTIONET_ENRICHMENT_HPP
