#ifndef ACTIONET_SPECIFICITY_EXT_H
#define ACTIONET_SPECIFICITY_EXT_H

#include "network.hpp"

arma::field<arma::mat> compute_feature_specificity_bin(arma::sp_mat &Sb, arma::mat &H, int thread_no = 0);

#endif //ACTIONET_SPECIFICITY_EXT_H
