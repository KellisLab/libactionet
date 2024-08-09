#ifndef ACTION_HPP
#define ACTION_HPP

#include "config_arma.hpp"
#include "config_interface.hpp"

// To store the output of run_SPA()
struct SPA_results
{
  arma::uvec selected_columns;
  arma::vec column_norms;
};

// To store the output of run_ACTION()
struct ACTION_results
{
  arma::field<arma::uvec> selected_cols;
  arma::field<arma::mat> H;
  arma::field<arma::mat> C;
};

// To store the output of prune_archetypes()
struct multilevel_archetypal_decomposition
{
  arma::uvec selected_archs; // If hub removal requested, this will hold the indices
                             // of retained archetypes
  arma::mat C_stacked;       // Stacking of C matrices, after potentially removing the hub
                             // archetypes
  arma::mat H_stacked;       // Stacking of H matrices, after potentially removing the hub
                             // archetypes
};

// To store the output of unify_archetypes()
struct unification_results
{
  arma::mat dag_adj;
  arma::vec dag_node_annotations;
  arma::uvec selected_archetypes;
  arma::mat C_unified;
  arma::mat H_unified;
  arma::uvec assigned_archetypes;
  arma::vec archetype_group;
  arma::mat arch_membership_weights;
};

namespace ACTIONet
{
  // svd
  // Basic (randomized) SVD algorithms
  arma::field<arma::mat> perturbedSVD(arma::field<arma::mat> SVD_results, arma::mat &A, arma::mat &B);

  arma::field<arma::mat> IRLB_SVD(arma::sp_mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> IRLB_SVD(arma::mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> FengSVD(arma::sp_mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> FengSVD(arma::mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> HalkoSVD(arma::sp_mat &A, int dim, int iters, int seed, int verbose);

  arma::field<arma::mat> HalkoSVD(arma::mat &A, int dim, int iters, int seed, int verbose);

  // reduction
  // Entry-points to compute a reduced kernel matrix
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

  // spa
  // Successive Projection Algorithm (SPA) to solve separable NMF
  SPA_results run_SPA(arma::mat &A, int k);

  // simplex_regression
  // Simplex regression ofr AA: min_{X} (|| AX - B ||) s.t. simplex constraint using ACTIVE Set Method
  arma::mat run_simplex_regression(arma::mat &A, arma::mat &B, bool computeXtX);

  // aa
  // Solves the standard Archetypal Analysis (AA) problem
  arma::field<arma::mat> run_AA(arma::mat &A, arma::mat &W0, int max_it, double min_delta);

  // action_decomp
  // Runs main ACTION decomposition
  ACTION_results run_ACTION(arma::mat &S_r, int k_min, int k_max, int thread_no, int max_it = 100, double min_delta = 1e-6, int normalization = 0);

  // action_post
  // Postprocessing ACTION results
  multilevel_archetypal_decomposition prune_archetypes(arma::field<arma::mat> C_trace, arma::field<arma::mat> H_trace, double min_specificity_z_threshold,
                                                       int min_cells = 3);

  unification_results unify_archetypes(arma::mat &S_r, arma::mat &C_stacked, arma::mat &H_stacked, double backbone_density, double resolution,
                                       int min_cluster_size, int thread_no, int normalization);

} // namespace ACTIONet

#endif
