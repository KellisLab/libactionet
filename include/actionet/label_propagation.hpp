// Label propagation algorithm (LPA)
#ifndef LIBACTIONET_LABEL_PROPAGATION_HPP
#define LIBACTIONET_LABEL_PROPAGATION_HPP

#include "libactionet_config.hpp"
#include "actionet/enrichment.hpp"
#include "utils_internal/utils_graph.hpp"
#include "utils_internal/utils_misc.hpp"

namespace ACTIONet {

    arma::vec LPA(arma::sp_mat &G, arma::vec labels, double lambda = 0, int iters = 3, double sig_threshold = 3,
                  arma::uvec fixed_labels = arma::uvec(), int thread_no = 0);

}

#endif //LIBACTIONET_LABEL_PROPAGATION_HPP
