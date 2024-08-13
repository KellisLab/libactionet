#ifndef LIBACTIONET_SPECIFICITY_EXT_H
#define LIBACTIONET_SPECIFICITY_EXT_H

#include "actionet.hpp"

arma::field<arma::mat> compute_feature_specificity_bin(arma::sp_mat &Sb, arma::mat &H, int thread_no = 0);

#endif //LIBACTIONET_SPECIFICITY_EXT_H
