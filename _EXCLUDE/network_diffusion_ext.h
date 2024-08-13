#ifndef LIBACTIONET_NETWORK_DIFFUSION_EXT_H
#define LIBACTIONET_NETWORK_DIFFUSION_EXT_H

#include "actionet.hpp"

arma::mat PR_linsys(arma::sp_mat &G, arma::sp_mat &X, double alpha = 0.85, int thread_no = -1);

arma::mat compute_network_diffusion_direct(arma::sp_mat &G, arma::sp_mat &X0, int thread_no = 4, double alpha = 0.85);

#endif //LIBACTIONET_NETWORK_DIFFUSION_EXT_H
