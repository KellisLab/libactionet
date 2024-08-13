#ifndef REDUCTION_EXT_H
#define REDUCTION_EXT_H

#include "action.hpp"

arma::field<arma::mat> SVD2ACTIONred(arma::sp_mat &S, arma::field<arma::mat> SVD_results);

arma::field<arma::mat> SVD2ACTIONred(arma::mat &S, arma::field<arma::mat> SVD_results);

arma::field<arma::mat> PCA2ACTIONred(arma::sp_mat &S, arma::field<arma::mat> PCA_results);

arma::field<arma::mat> PCA2ACTIONred(arma::mat &S, arma::field<arma::mat> PCA_results);

arma::field<arma::mat> ACTIONred2SVD(arma::field<arma::mat> SVD_results);

#endif //REDUCTION_EXT_H
