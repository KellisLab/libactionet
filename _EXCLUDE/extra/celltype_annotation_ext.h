#ifndef ACTIONET_CELLTYPE_ANNOTATION_EXT_HPP
#define ACTIONET_CELLTYPE_ANNOTATION_EXT_HPP

#include "network.hpp"

double F2z(double F, double d1, double d2);

arma::mat doubleNorm(arma::mat &X);

arma::sp_mat scale_expression(arma::sp_mat &S);

arma::mat aggregate_genesets(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                 int network_normalization_method = 0, int expression_normalization_method = 0,
                                 int gene_scaling_method = 0, int thread_no = 0);

arma::mat compute_marker_aggregate_stats_basic_sum(arma::sp_mat &S, arma::sp_mat &marker_mat);

arma::mat compute_marker_aggregate_stats_basic_sum_perm(arma::sp_mat &S, arma::sp_mat &marker_mat, int perm_no = 100,
                                                        int thread_no = 0);

arma::mat compute_marker_aggregate_stats_basic_sum_perm_smoothed(arma::sp_mat &G, arma::sp_mat &S,
                                                                 arma::sp_mat &marker_mat, double alpha = 0.85,
                                                                 int max_it = 5, int perm_no = 100, int thread_no = 0);

arma::mat compute_marker_aggregate_stats_basic_sum_smoothed(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                            double alpha = 0.85, int max_it = 5, int perm_no = 100,
                                                            int thread_no = 0);

arma::mat compute_marker_aggregate_stats_basic_sum_smoothed_normalized(arma::sp_mat &G, arma::sp_mat &S,
                                                                       arma::sp_mat &marker_mat, double alpha = 0.85,
                                                                       int max_it = 5, int perm_no = 100,
                                                                       int thread_no = 0);

arma::mat compute_marker_aggregate_stats_basic_sum_perm_smoothed_v2(arma::sp_mat &G, arma::sp_mat &S,
                                                                    arma::sp_mat &marker_mat, double alpha = 0.85,
                                                                    int max_it = 5, int perm_no = 100,
                                                                    int thread_no = 0);

arma::mat compute_marker_aggregate_stats_nonparametric(arma::mat &S, arma::sp_mat &marker_mat, int thread_no);

arma::mat compute_marker_aggregate_stats_TFIDF_sum_smoothed(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                            double alpha = 0.85, int max_it = 5, int perm_no = 100,
                                                            int thread_no = 0, int normalization = 1);

arma::mat aggregate_genesets_weighted_enrichment_permutation(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                             int network_normalization_method = 0,
                                                             int expression_normalization_method = 0,
                                                             int gene_scaling_method = 3, double pre_alpha = 0.15,
                                                             double post_alpha = 0.85, int thread_no = 0,
                                                             int perm_no = 30);

arma::mat aggregate_genesets_weighted_enrichment(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                 int network_normalization_method = 0,
                                                 int expression_normalization_method = 0, int gene_scaling_method = 3,
                                                 double pre_alpha = 0.15, double post_alpha = 0.85, int thread_no = 0);

arma::mat compute_markers_eigengene(arma::mat &S, arma::sp_mat &marker_mat, int normalization = 0, int thread_no = 0);

arma::field<arma::mat> aggregate_genesets_vision(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                 int network_normalization_method, double alpha, int thread_no);

#endif //ACTIONET_CELLTYPE_ANNOTATION_EXT_HPP
