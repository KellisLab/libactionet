#ifndef VISUALIZATION_EXT_H
#define VISUALIZATION_EXT_H

#include "visualization.hpp"

// Functions
arma::mat transform_layout(arma::sp_mat &G, arma::mat &reference_layout, bool presmooth_network,
                           const std::string &method, double min_dist, double spread, double gamma,
                           unsigned int n_epochs, int thread_no, int seed, double learning_rate, int sim2dist);

arma::sp_mat smoothKNN(arma::sp_mat &D, int max_iter = 64, double epsilon = 1e-6, double bandwidth = 1.0,
                       double local_connectivity = 1.0, double min_k_dist_scale = 1e-3, double min_sim = 1e-8,
                       int thread_no = 0);

#endif //VISUALIZATION_EXT_H
