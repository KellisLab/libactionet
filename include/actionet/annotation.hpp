#ifndef LIBACTIONET_ANNOTATION_HPP
#define LIBACTIONET_ANNOTATION_HPP

#include "libactionet_config.hpp"
#include "actionet/network_diffusion.hpp"
#include "utils_internal/utils_network.hpp"
#include "action/aa.hpp"
#include "tools/tl_normalization.hpp"

namespace ACTIONet {

    arma::mat compute_marker_aggregate_stats(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                             double alpha = 0.85, int max_it = 5, int thread_no = 0,
                                             bool ignore_baseline_expression = false);

    arma::mat aggregate_genesets(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                 int network_normalization_method = 0, int expression_normalization_method = 0,
                                 int gene_scaling_method = 0, double post_alpha = 0.85, int thread_no = 0);

    arma::mat aggregate_genesets_mahalanobis_2archs(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                    int network_normalization_method = 0,
                                                    int expression_normalization_method = 0,
                                                    int gene_scaling_method = 3,
                                                    double pre_alpha = 0.15, double post_alpha = 0.85,
                                                    int thread_no = 0);

    arma::mat aggregate_genesets_mahalanobis_2gmm(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                  int network_normalization_method = 0,
                                                  int expression_normalization_method = 0, int gene_scaling_method = 3,
                                                  double pre_alpha = 0.15, double post_alpha = 0.85, int thread_no = 0);

}

#endif //LIBACTIONET_ANNOTATION_HPP
