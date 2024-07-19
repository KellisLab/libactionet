#ifndef SVD_HPP
#define SVD_HPP

#include "action.hpp"

// #include "config_actionet.hpp"

#include "helpers_matrix.hpp"
#include "aarand/aarand.hpp"

#include <cblas.h>

void orthog(double *X, double *Y, double *T, int xm, int xn, int yn);

void convtests(int Bsz, int n, double tol, double svtol, double Smax, double *svratio,
               double *residuals, int *k, int *converged, double S);

arma::mat randNorm(int l, int m, int seed);

arma::field<arma::mat> eigSVD(arma::mat A);

void gram_schmidt(arma::mat &A);

arma::field<arma::mat> orient_SVD(arma::field<arma::mat> SVD_res);

#endif
