#ifndef LIBACTIONET_HDBSCAN_H
#define LIBACTIONET_HDBSCAN_H

#include "libactionet_config.hpp"

arma::field<arma::vec> run_HDBSCAN(arma::mat &X, int minPoints = 5, int minClusterSize = 5);

#endif //LIBACTIONET_HDBSCAN_H
