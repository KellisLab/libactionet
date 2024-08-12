#ifndef AA_EXT_H
#define AA_EXT_H

#include "action.hpp"

// To store the output of compute_AA_coreset()
struct Coreset {
    arma::mat S_coreset;
    arma::vec w_coreset;
    arma::uvec index;
};

arma::field<arma::mat>
run_weighted_AA(arma::mat &A, arma::mat &W0, arma::vec w, int max_it = 50, double min_delta = 0.01);

arma::field<arma::mat> Online_update_AA(arma::mat &Xt, arma::mat &D, arma::mat &A, arma::mat &B);

arma::field<arma::mat> run_online_AA(arma::mat &X, arma::mat &D0, arma::field<arma::uvec> samples);

arma::field<arma::mat> run_AA_with_prior(arma::mat &A, arma::mat &W0, arma::mat &W_prior, int max_it = 50,
                                         double min_delta = 1e-16);

Coreset compute_AA_coreset(arma::sp_mat &S, int m = 5000);

#endif //AA_EXT_H
