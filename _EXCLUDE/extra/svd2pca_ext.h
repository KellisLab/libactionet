#ifndef ACTIONET_SVD2PCA_HPP
#define ACTIONET_SVD2PCA_HPP

#include "libactionet_config.hpp"

template<typename T>
arma::field<arma::mat> SVD2PCA(T &S, arma::field<arma::mat> SVD_results);

template<typename T>
arma::field<arma::mat> PCA2SVD(T &S, arma::field<arma::mat> PCA_results);

#endif //ACTIONET_SVD2PCA_HPP
