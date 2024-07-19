#ifndef ACTION_HPP
#define ACTION_HPP

#include "config_arma.hpp"
#include "config_actionet.hpp"

namespace ACTIONet
{
  // simplex_regression
  // Successive Projection Algorithm (SPA) to solve separable NMF
  arma::mat run_simplex_regression(arma::mat &A, arma::mat &B, bool computeXtX);

  // svd
  // Basic (randomized) SVD algorithms
  arma::field<arma::mat> IRLB_SVD(arma::sp_mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> IRLB_SVD(arma::mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> FengSVD(arma::sp_mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> FengSVD(arma::mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> HalkoSVD(arma::sp_mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> HalkoSVD(arma::mat &A, int dim, int iters, int seed, int verbose);

  // reduction
  // Entry-points to compute a reduced kernel matrix
  arma::field<arma::mat> perturbedSVD(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B);

  arma::field<arma::mat> PCA2SVD(arma::sp_mat &S, arma::field<arma::mat> PCA_results);

  arma::field<arma::mat> PCA2SVD(arma::mat &S, arma::field<arma::mat> PCA_results);

  arma::field<arma::mat> SVD2PCA(arma::sp_mat &S, arma::field<arma::mat> SVD_results);

  arma::field<arma::mat> SVD2PCA(arma::mat &S, arma::field<arma::mat> SVD_results);

  arma::field<arma::mat> reduce_kernel(arma::sp_mat &S, int dim, int iter, int seed, int SVD_algorithm, bool prenormalize, int verbose);

  arma::field<arma::mat> reduce_kernel(arma::mat &S, int dim, int iter, int seed, int SVD_algorithm, bool prenormalize, int verbose);

  // orthogonalization
  // Batch orthogonalization interface
  arma::field<arma::mat> orthogonalize_batch_effect(arma::sp_mat &S, arma::field<arma::mat> SVD_results, arma::mat &design);

  arma::field<arma::mat> orthogonalize_batch_effect(arma::mat &S, arma::field<arma::mat> SVD_results, arma::mat &design);

  arma::field<arma::mat> orthogonalize_basal(arma::sp_mat &S, arma::field<arma::mat> SVD_results, arma::mat &basal);

  arma::field<arma::mat> orthogonalize_basal(arma::mat &S, arma::field<arma::mat> SVD_results, arma::mat &basal);

} // namespace ACTIONet

#endif
