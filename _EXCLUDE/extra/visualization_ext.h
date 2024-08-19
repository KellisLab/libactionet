#ifndef VISUALIZATION_EXT_H
#define VISUALIZATION_EXT_H

#include "visualization.hpp"

// Functions
arma::mat transform_layout(arma::sp_mat &G, arma::mat &reference_layout, bool presmooth_network,
                           const std::string &method, double min_dist, double spread, double gamma,
                           unsigned int n_epochs, int thread_no, int seed, double learning_rate, int sim2dist);

#endif //VISUALIZATION_EXT_H
