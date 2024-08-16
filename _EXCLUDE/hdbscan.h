#ifndef ACTIONET_HDBSCAN_H
#define ACTIONET_HDBSCAN_H

#include "libactionet_config.hpp"

arma::field<arma::vec> run_HDBSCAN(arma::mat &X, int minPoints = 5, int minClusterSize = 5);

#endif //ACTIONET_HDBSCAN_H
