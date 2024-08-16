#ifndef ACTIONET_SVD_PRIV_HPP
#define ACTIONET_SVD_PRIV_HPP

#include "libactionet_config.hpp"
#include "utils_internal/utils_matrix.hpp"
#include "aarand/aarand.hpp"

// Functions: private
void orthog(double *X, double *Y, double *T, int xm, int xn, int yn);

void convtests(int Bsz, int n, double tol, double svtol, double Smax, double *svratio, double *residuals, int *k,
               int *converged, double S);

arma::mat randNorm(int l, int m, int seed);

arma::field <arma::mat> eigSVD(arma::mat A);

// Functions: internal
void gram_schmidt(arma::mat &A);

#endif //ACTIONET_SVD_PRIV_HPP
