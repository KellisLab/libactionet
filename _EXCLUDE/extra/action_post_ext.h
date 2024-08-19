#ifndef ACTION_POST_EXT_H
#define ACTION_POST_EXT_H

#include "action.hpp"

double Kappa(double p, double q);

double log_HGT_tail(int population_size, int success_count, int sample_size,
                    int observed_success);

double assess_overlap(arma::uvec i1, arma::uvec i2, int population_size);

arma::mat compute_overlap_matrix(arma::mat C);

arma::mat NetEnh(arma::mat A);

arma::field<arma::mat> nndsvd(arma::mat &A, int dim = 100, int max_iter = 5);

// Redundant with `orient_SVD` in "svd.hpp"
arma::field<arma::mat> orient_SVD(arma::field<arma::mat> &SVD_out);

arma::field<arma::mat> convexSVD(arma::mat &A, int dim = 100, int max_iter = 5);

arma::field<arma::mat> recursiveNMU(arma::mat M, int dim = 100, int max_SVD_iter = 5,
                                    int max_iter_inner = 40);

arma::field<arma::mat> recursiveNMU_mine(arma::mat M, int dim = 100, int max_SVD_iter = 1000,
                                         int max_iter_inner = 100);

arma::vec sweepcut(arma::sp_mat &A, arma::vec s, int min_size, int max_size);

#endif //ACTION_POST_EXT_H
