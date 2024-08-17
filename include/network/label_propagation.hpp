// Label propagation algorithm (LPA)
#ifndef ACTIONET_LABEL_PROPAGATION_HPP
#define ACTIONET_LABEL_PROPAGATION_HPP

#include "libactionet_config.hpp"

namespace actionet {

    arma::vec LPA(arma::sp_mat &G, arma::vec labels, double lambda = 0, int iters = 3, double sig_threshold = 3,
                  arma::uvec fixed_labels = arma::uvec(), int thread_no = 0);

}

#endif //ACTIONET_LABEL_PROPAGATION_HPP
