#ifndef ACTIONET_NETWORK_DIFFUSION_EXT_H
#define ACTIONET_NETWORK_DIFFUSION_EXT_H

#include "network.hpp"

arma::mat PR_linsys(arma::sp_mat &G, arma::sp_mat &X, double alpha = 0.85, int thread_no = -1);

arma::mat compute_network_diffusion_direct(arma::sp_mat &G, arma::sp_mat &X0, int thread_no = 4, double alpha = 0.85);

#endif //ACTIONET_NETWORK_DIFFUSION_EXT_H
