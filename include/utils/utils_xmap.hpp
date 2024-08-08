#ifndef UTILS_XMAP_HPP
#define UTILS_XMAP_HPP

#include "config_arma.hpp"
#include <cfloat>

// Constants
#define NEGATIVE_SAMPLE_RATE 3.0
#define UMAP_SEED 0
#define GAMMA 1.0
#define ADAM_ALPHA 1.0 /*same as learning_rate*/
#define ADAM_BETA1 0.5 /*only adam: between 0 and 1*/
#define ADAM_BETA2 0.9 /*only adam: between 0 and 1*/
#define ADAM_EPS 1e-7  /*only adam: between 1e-8 and 1e-3*/

// Functions
arma::sp_mat smoothKNN(arma::sp_mat &D, int max_iter = 64, double epsilon = 1e-6, double bandwidth = 1.0,
                       double local_connectivity = 1.0, double min_k_dist_scale = 1e-3, double min_sim = 1e-8,
                       int thread_no = 0);

#endif
