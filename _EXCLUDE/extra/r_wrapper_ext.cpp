#include "libactionet_config.hpp"
#include "action.hpp"
#include "network.hpp"
#include "visualization.hpp"
#include "tools.hpp"

//' Set the RNG Seed from within Rcpp
//'
//' Within Rcpp, one can set the R session seed without triggering
//' the CRAN rng modifier check.
//' @param seed A \code{double} that is the seed one wishes to use.
//' @return A set RNG scope.
//' @examples
//' set.seed(10)
//' x = rnorm(5,0,1)
//' set_seed(10)
//' y = rnorm(5,0,1)
//' all.equal(x,y, check.attributes = F)
void set_seed(double seed) {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(std::floor(std::fabs(seed)));
}

Rcpp::List
runACTION_muV(const Rcpp::List &S, int k_min, int k_max, arma::vec alpha, double lambda = 1, int AA_iters = 50,
               int Opt_iters = 0, int thread_no = 0) {

    int n_list = S.size();
    std::vector<arma::mat> cell_signatures(n_list);
    for (int i = 0; i < n_list; i++) {
        cell_signatures[i] = (as<arma::mat>(S[i]));
    }

    full_trace run_trace = runACTION_muV(cell_signatures, k_min, k_max, alpha, lambda, AA_iters, Opt_iters, thread_no);

    Rcpp::List res;

    Rcpp::List H_consensus(k_max);
    for (int kk = k_min; kk <= k_max; kk++) {
        H_consensus[kk - 1] = run_trace.H_consensus[kk];
    }
    res["H_consensus"] = H_consensus;

    char ds_name[128];
    for (int i = 0; i < n_list; i++) {
        Rcpp::List individual_trace;

        Rcpp::List H_primary(k_max);
        for (int kk = k_min; kk <= k_max; kk++) {
            H_primary[kk - 1] = run_trace.indiv_trace[kk].H_primary[i];
        }
        individual_trace["H_primary"] = H_primary;

        Rcpp::List H_secondary(k_max);
        for (int kk = k_min; kk <= k_max; kk++) {
            H_secondary[kk - 1] = run_trace.indiv_trace[kk].H_secondary[i];
        }
        individual_trace["H_secondary"] = H_secondary;

        Rcpp::List C_primary(k_max);
        for (int kk = k_min; kk <= k_max; kk++) {
            C_primary[kk - 1] = run_trace.indiv_trace[kk].C_primary[i];
        }
        individual_trace["C_primary"] = C_primary;

        Rcpp::List C_consensus(k_max);
        for (int kk = k_min; kk <= k_max; kk++) {
            C_consensus[kk - 1] = run_trace.indiv_trace[kk].C_consensus[i];
        }
        individual_trace["C_consensus"] = C_consensus;

        sprintf(ds_name, "View%d_trace", i + 1);
        res[ds_name] = individual_trace;
    }

    return res;
}

//' Runs Successive Projection Algorithm (SPA) to solve separable NMF
//'
//' @param A Input matrix
//' @param k Number of columns to select
//'
//' @return A named list with entries 'selected_cols' and 'norms'
//' @examples
//' H = runSPA(S_r, 10)
Rcpp::List runSPA_rows_sparse(arma::sp_mat &A, int k) {
    actionet::ResSPA res = runSPA_rows_sparse(A, k);
    arma::uvec selected_cols = res.selected_cols;

    arma::vec cols(k);
    for (int i = 0; i < k; i++) {
        cols[i] = selected_cols[i] + 1;
    }

    Rcpp::List out;
    out["selected_rows"] = cols;
    out["norms"] = res.column_norms;

    return out;
}

//' Runs multi-level ACTION decomposition method
//'
//' @param S_r Reduced kernel matrix
//' @param k_min Minimum number of archetypes to consider (default=2)
//' @param k_max Maximum number of archetypes to consider, or "depth" of
//' decomposition (default=30)
//' @param max_it,min_delta Convergence parameters for archetypal analysis
//' @param max_trial Maximum number of trials before termination
//'
//' @return A named list with entries 'C' and 'H', each a list for different
//' values of k ' @examples ' ACTION.out = runACTION_plus(S_r, k_max = 10) ' H8
//' = ACTION.out$H[[8]] ' cell.assignments = apply(H8, 2, which.max)
Rcpp::List runACTION_plus(arma::mat &S_r, int k_min = 2, int k_max = 30, int max_it = 100, double min_delta = 1e-6,
                           int max_trial = 3) {

    ResACTION trace = runACTION_plus(S_r, k_min, k_max, max_it, min_delta, max_trial);

    Rcpp::List res;

    Rcpp::List C(trace.H.n_elem - 1);
    for (int i = k_min; i < trace.H.n_elem; i++) {
        C[i - 1] = trace.C[i];
    }
    res["C"] = C;

    Rcpp::List H(trace.H.n_elem - 1);
    for (int i = k_min; i < trace.H.n_elem; i++) {
        H[i - 1] = trace.H[i];
    }
    res["H"] = H;

    return res;
}

//' Runs multi-level Online ACTION decomposition method (under development)
//'
//' @param S_r Reduced kernel matrix
//' @param k_min Minimum number of archetypes to consider (default=2)
//' @param k_max Maximum number of archetypes to consider, or "depth" of
//' decomposition (default=30)
//' @param samples list of sampled cells to use for updating archetype decomposition
//' @param thread_no Number of parallel threads (default = 0)
//'
//' @return A named list with entries 'C' and 'H', each a list for different
//' values of k ' @examples ' ACTION.out = run_online_ACTION(S_r, k_max = 10)
Rcpp::List
run_online_ACTION(arma::mat &S_r, arma::field<arma::uvec> samples, int k_min = 2, int k_max = 30, int thread_no = 0) {

    Online_ACTION_results trace = run_online_ACTION(S_r, samples, k_min, k_max, thread_no);

    Rcpp::List res;

    Rcpp::List A(k_max);
    for (int i = k_min; i <= k_max; i++) {
        A[i - 1] = trace.A[i];
    }
    res["A"] = A;

    Rcpp::List B(k_max);
    for (int i = k_min; i <= k_max; i++) {
        B[i - 1] = trace.B[i];
    }
    res["B"] = B;

    Rcpp::List C(k_max);
    for (int i = k_min; i <= k_max; i++) {
        C[i - 1] = trace.C[i];
    }
    res["C"] = C;

    Rcpp::List D(k_max);
    for (int i = k_min; i <= k_max; i++) {
        D[i - 1] = trace.D[i];
    }
    res["D"] = D;

    return res;
}

//' Runs multi-level weighted ACTION decomposition method (under development)
//'
//' @param S_r Reduced kernel matrix
//' @param w Weight vector for each observation
//' @param k_min Minimum number of archetypes to consider (default=2)
//' @param k_max Maximum number of archetypes to consider, or "depth" of
//' decomposition (default=30)
//' @param thread_no Number of parallel threads (default=0)
//'
//' @return A named list with entries 'C' and 'H', each a list for different
//' values of k
//'
//' @examples ACTION.out = run_weighted_ACTION(S_r, w, k_max = 20)
Rcpp::List run_weighted_ACTION(arma::mat &S_r, arma::vec w, int k_min = 2, int k_max = 30, int thread_no = 0,
                               int max_it = 50, double min_delta = 1e-16) {

    ResACTION trace = run_weighted_ACTION(S_r, w, k_min, k_max, thread_no, max_it, min_delta);

    Rcpp::List res;

    Rcpp::List C(k_max);
    for (int i = k_min; i <= k_max; i++) {
        C[i - 1] = trace.C[i];
    }
    res["C"] = C;

    Rcpp::List H(k_max);
    for (int i = k_min; i <= k_max; i++) {
        H[i - 1] = trace.H[i];
    }
    res["H"] = H;

    return res;
}

//' Renormalized input matrix to minimize differences in means
//'
//' @param S Input matrix
//' @param sample_assignments Any primary grouping - typically based on cell
//' type/state (it has to be in {1, ..., k1})
//'
//' @return A list with the first entry being the renormalized input matrix
//'
//' @examples
//' prune.out = collectArchetypes(ACTION.out$C, ACTION.out$H)
//'	G = buildNetwork(prune.out$H_stacked)
//' unification.out = mergeArchetypes(G, S_r, prune.out$C_stacked, prune.out$H_stacked)
//' cell.clusters = unification.out$sample_assignments
//' S.norm = renormalize_input_matrix(S, cell.clusters)
arma::sp_mat renormalize_input_matrix(arma::sp_mat &S, arma::Col<unsigned long long> sample_assignments) {

    arma::sp_mat S_norm = renormalize_input_matrix(S, sample_assignments);

    return (S_norm);
}

arma::mat renormalize_input_matrix_full(arma::mat &S, arma::Col<unsigned long long> sample_assignments) {

    arma::mat S_norm = renormalize_input_matrix(S, sample_assignments);

    return (S_norm);
}

//' Compute feature specificity (from archetype footprints and binary input)
//'
//' @param S Input matrix (sparseMatrix - binary)
//' @param H A soft membership matrix - Typically H_merged from the mergeArchetypes() function.
//'
//' @return A list with the over/under-logPvals
//'
//' @examples
//'	logPvals.list = compute_archetype_feature_specificity_bin(S.bin, unification.out$H_merged)
//' specificity.scores = logPvals.list$upper_significance
Rcpp::List compute_archetype_feature_specificity_bin(arma::sp_mat &S, arma::mat &H, int thread_no = 0) {

    arma::field<arma::mat> res = compute_feature_specificity_bin(S, H, thread_no);

    Rcpp::List out_list;
    out_list["archetypes"] = res(0);
    out_list["upper_significance"] = res(1);
    out_list["lower_significance"] = res(2);

    return (out_list);
}

//' Computes network diffusion over a given network, starting with an arbitrarty
//' set of initial scores (direct approach)
//'
//' @param G Input graph
//' @param X0 Matrix of initial values per diffusion (ncol(G) == nrow(G) == ncol(X0))
//' @param thread_no Number of parallel threads (default=0)
//' @param alpha Random-walk depth ( between [0, 1] ) ' @param max_it PageRank iterations
//'
//' @return Matrix of diffusion scores
//'
//' @examples
//' G = colNets(ace)$ACTIONet
//' gene.expression = Matrix::t(logcounts(ace))[c("CD19", "CD14", "CD16"), ]
//' smoothed.expression = compute_network_diffusion_direct(G, gene.expression)
arma::mat compute_network_diffusion_direct(arma::sp_mat &G, arma::sp_mat &X0, int thread_no = 0, double alpha = 0.85) {
    arma::mat Diff = compute_network_diffusion_direct(G, X0, thread_no, alpha);

    return (Diff);
}

//' Computes disjoint clusters for vertices of G.
//' (It uses an adjusted DBSCAN procedure)
//'
//' @param G Adjacency matrix of the input graph
//' @param minPts, eps DBSCAN parameters
//' @param alpha Diffusion parameter for initial node ordering
//'
//' @return Matrix of log-pvalues
//'
//' @examples
//' G = colNets(ace)$ACTIONet
//' clusters = NetDBSCAN(G)
arma::vec NetDBSCAN(SEXP G, int minPts = 10, double eps = 0.5, double alpha = 0.85) {
    arma::sp_mat Adj;
    if (Rf_isS4(G)) {
        Adj = Rcpp::as<arma::sp_mat>(G);
    } else {
        Adj = arma::sp_mat(Rcpp::as<arma::mat>(G));
    }

    arma::vec clusters = NetDBSCAN(Adj, minPts, eps, alpha);

    return (clusters);
}

//' Clusters data points using the hierarchical DBSCAN algorithm.
//'
//' @param X Input data matrix with each row being a data point
//'
//' @return A list with \itemize{
//' \item labels
//' \item membershipProbabilities
//' \item outlierScores
//' }
//'
//' @examples
//' S_r = t(reducedDims(ace)[["S_r"]])
//' W_r = S_r %*% trace$pruning.out$C_stacked
//' X = Matrix::t(W_r)
//' HDBSCAN.out = run_HDBSCAN(X)
//' clusters = HDBSCAN.out$labels
Rcpp::List run_HDBSCAN(arma::mat &X, int minPoints = 5, int minClusterSize = 5) {
    arma::field<arma::vec> res = run_HDBSCAN(X, minPoints, minClusterSize);

    Rcpp::List out_list;
    out_list["labels"] = res(0);
    out_list["membershipProbabilities"] = res(1);
    out_list["outlierScores"] = res(2);

    return (out_list);
}

//' Computes a coreset for archetypal analysis
//' Ref: Coresets for Archetypal Analysis:
//' (http://papers.neurips.cc/paper/8945-coresets-for-archetypal-analysis)
//'
//' @param S Input matrix (e.g., gene x cell)
//' @param m Number of samples (or 0, to be automatically identified)
//' @param seed Random seed
//'
//' @return clusters Assignment vector of samples to clusters
//'
//' @examples
//' coreset = compute_AA_coreset(S, 1000)
Rcpp::List compute_AA_coreset(arma::sp_mat &S, int m = 0) {
    actionet::Coreset coreset = compute_AA_coreset(S, m);

    Rcpp::List out_list;
    out_list["S_coreset"] = coreset.S_coreset;
    out_list["w_coreset"] = coreset.w_coreset;

    arma::uvec index = coreset.index + 1;
    out_list["index"] = index;

    return (out_list);
}

//' Computes reduced kernel matrix for a given (single-cell) profile and prior
//' SVD
//'
//' @param S Input matrix ("sparseMatrix")
//' @param U Left singular vectors
//' @param s signular values
//' @param V Right singular vectors
//'
//' @return A named list with S_r, V, lambda, and exp_var.
//' \itemize{
//' \item S_r: reduced kernel matrix of size reduced_dim x #samples.
//' \item V: Associated left singular-vectors (useful for reconstructing
//' discriminative scores for features, such as genes).
//' \item lambda, exp_var: Summary statistics of the sigular-values.
//' }
//'
//' @examples
//' S = logcounts(sce)
//' irlba.out = irlba::irlba(S, nv = 50)
//' red.out = SVD2ACTIONred_full(S, irlba.out$u, as.matrix(irlba.out$d), irlba.out$v)
//' Sr = red.out$S_r
Rcpp::List SVD2ACTIONred(arma::sp_mat &S, arma::mat u, arma::vec d, arma::mat v) {
    if (1 < d.n_cols)
        d = d.diag();

    arma::field<arma::mat> SVD_results(3);
    SVD_results(0) = u;
    SVD_results(1) = d;
    SVD_results(2) = v;

    arma::field<arma::mat> reduction = SVD2ACTIONred(S, SVD_results);

    Rcpp::List res;
    res["V"] = reduction(0);

    arma::vec sigma = reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);

    res["A"] = reduction(3);
    res["B"] = reduction(4);

    return res;
}

Rcpp::List SVD2ACTIONred_full(arma::mat &S, arma::mat u, arma::vec d, arma::mat v) {
    if (1 < d.n_cols)
        d = d.diag();

    arma::field<arma::mat> SVD_results(3);
    SVD_results(0) = u;
    SVD_results(1) = d;
    SVD_results(2) = v;

    arma::field<arma::mat> reduction = SVD2ACTIONred(S, SVD_results);

    Rcpp::List res;
    res["V"] = reduction(0);

    arma::vec sigma = reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);

    res["A"] = reduction(3);
    res["B"] = reduction(4);

    return res;
}

//' Computes reduced kernel matrix for a given (single-cell) profile and prior
//' SVD
//'
//' @param S Input matrix ("sparseMatrix")
//' @param U Left singular vectors
//' @param s signular values
//' @param V Right singular vectors
//'
//' @return A named list with S_r, V, lambda, and exp_var.
//' \itemize{
//' \item S_r: reduced kernel matrix of size reduced_dim x #samples.
//' \item V: Associated left singular-vectors (useful for reconstructing
//' discriminative scores for features, such as genes).
//' \item lambda, exp_var: Summary statistics of the sigular-values.
//' }
//'
//' @examples
//' S = logcounts(sce)
//' irlba.out = irlba::prcomp_irlba(S, n = 50, retx = TRUE, center = T)
//' red.out = PCA2ACTIONred_full(S, irlba.out$x, irlba.out$rotation, as.matrix(irlba.out$sdev))
//' Sr = red.out$S_r
Rcpp::List PCA2ACTIONred(arma::sp_mat &S, arma::mat x, arma::vec sdev, arma::mat rotation) {
    arma::field<arma::mat> SVD_results(3);

    arma::vec d = sdev * std::sqrt(x.n_rows - 1);
    arma::mat U = x;
    for (int i = 0; i < U.n_cols; i++) {
        U.col(i) /= d(i);
    }

    SVD_results(0) = U;
    SVD_results(1) = d;
    SVD_results(2) = rotation;

    arma::field<arma::mat> reduction = PCA2ACTIONred(S, SVD_results);

    Rcpp::List res;
    res["V"] = reduction(0);

    arma::vec sigma = reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);
    res["A"] = reduction(3);
    res["B"] = reduction(4);

    return res;
}

Rcpp::List PCA2ACTIONred_full(arma::mat &S, arma::mat x, arma::vec sdev, arma::mat rotation) {
    arma::field<arma::mat> SVD_results(3);

    arma::vec d = sdev * std::sqrt(x.n_rows - 1);
    arma::mat U = x;
    for (int i = 0; i < U.n_cols; i++) {
        U.col(i) /= d(i);
    }

    SVD_results(0) = U;
    SVD_results(1) = d;
    SVD_results(2) = rotation;

    arma::field<arma::mat> reduction = PCA2ACTIONred(S, SVD_results);

    Rcpp::List res;
    res["V"] = reduction(0);

    arma::vec sigma = reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);
    res["A"] = reduction(3);
    res["B"] = reduction(4);

    return res;
}

Rcpp::List run_subACTION(arma::mat &S_r, arma::mat &W_parent, arma::mat &H_parent, int kk, int k_min, int k_max,
                         int thread_no, int max_it = 50, double min_delta = 1e-16) {
    actionet::ResACTION trace =
            run_subACTION(S_r, W_parent, H_parent, kk - 1, k_min, k_max, thread_no, max_it, min_delta);

    Rcpp::List res;

    Rcpp::List C(k_max);
    for (int i = k_min; i <= k_max; i++) {
        C[i - 1] = trace.C[i];
    }
    res["C"] = C;

    Rcpp::List H(k_max);
    for (int i = k_min; i <= k_max; i++) {
        H[i - 1] = trace.H[i];
    }
    res["H"] = H;

    return res;
}

Rcpp::List deflateReduction(arma::mat &old_S_r, arma::mat &old_V, arma::mat &old_A, arma::mat &old_B,
                             arma::vec &old_sigma, arma::mat &A, arma::mat &B) {
    arma::field<arma::mat> SVD_results(5);

    SVD_results(0) = old_V;
    SVD_results(1) = old_sigma;
    SVD_results(2) = old_S_r;
    for (int i = 0; i < old_sigma.n_elem; i++) {
        SVD_results(2).col(i) /= old_sigma(i);
    }
    SVD_results(3) = old_A;
    SVD_results(4) = old_B;

    arma::field<arma::mat> deflated_reduction = deflateReduction(SVD_results, A, B);

    Rcpp::List res;
    res["V"] = deflated_reduction(0);

    arma::vec sigma = deflated_reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = deflated_reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);
    res["A"] = deflated_reduction(3);
    res["B"] = deflated_reduction(4);

    return res;
}

arma::mat NetEnh(arma::mat A) {
    arma::mat A_enh = NetEnh(A);

    return (A_enh);
}

arma::mat compute_marker_aggregate_stats_TFIDF_sum_smoothed(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                            double alpha = 0.85, int max_it = 5, int perm_no = 100,
                                                            int thread_no = 0, int normalization = 1) {

    arma::mat stats = compute_marker_aggregate_stats_TFIDF_sum_smoothed(G, S, marker_mat, alpha, max_it, perm_no,
                                                                        thread_no, normalization);

    return (stats);
}

arma::mat transform_layout(arma::sp_mat &G, arma::mat reference_coordinates, const std::string &method = "umap",
                           bool presmooth_network = false, double min_dist = 1, double spread = 1, double gamma = 1.0,
                           unsigned int n_epochs = 500, int thread_no = 0, int seed = 0, double learning_rate = 1.0,
                           int sim2dist = 2) {

    arma::mat coors = transform_layout(G, reference_coordinates, presmooth_network, method, min_dist, spread, gamma,
                                       n_epochs, thread_no, seed, learning_rate, sim2dist);

    return (coors);
}

arma::mat compute_marker_aggregate_stats_nonparametric(arma::mat &S, arma::sp_mat &marker_mat, int thread_no = 0) {
    arma::mat X = compute_marker_aggregate_stats_nonparametric(S, marker_mat, thread_no);
    return (X);
}

arma::mat compute_markers_eigengene(arma::mat &S, arma::sp_mat &marker_mat, int normalization = 0, int thread_no = 0) {
    arma::mat X = compute_markers_eigengene(S, marker_mat, normalization, thread_no);
    return (X);
}

arma::vec sweepcut(arma::sp_mat &A, arma::vec s, int min_size = 5, int max_size = -1) {
    arma::vec cond = sweepcut(A, s, min_size, max_size);

    return (cond);
}

arma::sp_mat buildNetwork_bipartite(arma::mat H1, arma::mat H2, double density = 1.0, int thread_no = 0, double M = 16,
                                    double ef_construction = 200, double ef = 200, std::string distance_metric = "jsd") {
    arma::sp_mat G = buildNetwork_bipartite(H1, H2, density, thread_no, M, ef_construction, ef, distance_metric);

    return (G);
}

Rcpp::List recursiveNMU(arma::mat M, int dim = 100, int max_SVD_iter = 1000, int max_iter_inner = 100) {
    arma::field<arma::mat> stats = recursiveNMU(M, dim, max_SVD_iter, max_iter_inner);

    Rcpp::List res;
    res["W"] = stats[0];
    res["H"] = arma::trans(stats[1]);
    res["factor_weights"] = stats[2];

    return (res);
}

Rcpp::List recursiveNMU_mine(arma::mat M, int dim = 100, int max_SVD_iter = 1000, int max_iter_inner = 100) {
    arma::field<arma::mat> stats = recursiveNMU_mine(M, dim, max_SVD_iter, max_iter_inner);

    Rcpp::List res;
    res["W"] = stats[0];
    res["H"] = arma::trans(stats[1]);
    res["factor_weights"] = stats[2];

    return (res);
}

arma::mat aggregate_genesets_weighted_enrichment_permutation(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                             int network_normalization_method = 0,
                                                             int expression_normalization_method = 0,
                                                             int gene_scaling_method = 3, double pre_alpha = 0.85,
                                                             double post_alpha = 0.85, int thread_no = 0,
                                                             int perm_no = 30) {
    arma::mat stats =
            aggregate_genesets_weighted_enrichment_permutation(G, S, marker_mat, network_normalization_method,
                                                               expression_normalization_method, gene_scaling_method,
                                                               pre_alpha, post_alpha, thread_no, perm_no);

    return (stats);
}

arma::mat aggregate_genesets_weighted_enrichment(arma::sp_mat &G, arma::sp_mat &S, arma::sp_mat &marker_mat,
                                                 int network_normalization_method = 0,
                                                 int expression_normalization_method = 0, int gene_scaling_method = 3,
                                                 double pre_alpha = 0.85, double post_alpha = 0.85, int thread_no = 0) {

    arma::mat stats = aggregate_genesets_weighted_enrichment(G, S, marker_mat, network_normalization_method,
                                                             expression_normalization_method, gene_scaling_method,
                                                             pre_alpha, post_alpha, thread_no);

    return (stats);
}

Rcpp::List PCA2SVD(arma::sp_mat &S, arma::mat x, arma::vec sdev, arma::mat rotation) {
    arma::field<arma::mat> PCA_results(3);
    arma::vec d = sdev * std::sqrt(x.n_rows - 1);

    arma::mat U = x;
    for (int i = 0; i < U.n_cols; i++) {
        U.col(i) /= d(i);
    }
    PCA_results(0) = U;
    PCA_results(1) = d;
    PCA_results(2) = rotation;

    arma::field<arma::mat> SVD_results = actionet::PCA2SVD(S, PCA_results);

    Rcpp::List res;
    res["u"] = SVD_results(0);
    res["d"] = SVD_results(1);
    res["v"] = SVD_results(2);

    return res;
}

Rcpp::List PCA2SVD_full(arma::mat &S, arma::mat x, arma::vec sdev, arma::mat rotation) {
    arma::field<arma::mat> PCA_results(3);
    arma::vec d = sdev * std::sqrt(x.n_rows - 1);

    arma::mat U = x;
    for (int i = 0; i < U.n_cols; i++) {
        U.col(i) /= d(i);
    }
    PCA_results(0) = U;
    PCA_results(1) = d;
    PCA_results(2) = rotation;

    arma::field<arma::mat> SVD_results = actionet::PCA2SVD(S, PCA_results);

    Rcpp::List res;
    res["u"] = SVD_results(0);
    res["d"] = SVD_results(1);
    res["v"] = SVD_results(2);

    return res;
}

Rcpp::List SVD2PCA(arma::sp_mat &S, arma::mat u, arma::vec d, arma::mat v) {
    if (1 < d.n_cols)
        d = d.diag();

    arma::field<arma::mat> SVD_results(3);
    SVD_results(0) = u;
    SVD_results(1) = d;
    SVD_results(2) = v;

    arma::field<arma::mat> PCA_results = actionet::SVD2PCA(S, SVD_results);

    Rcpp::List res;
    arma::vec s = PCA_results(1).col(0);

    arma::mat X = PCA_results(0);
    for (int i = 0; i < X.n_cols; i++) {
        X.col(i) *= s(i);
    }
    res["x"] = X;
    res["rotation"] = PCA_results(2);
    res["sdev"] = s / std::sqrt(X.n_rows - 1);

    return res;
}

Rcpp::List SVD2PCA_full(arma::mat &S, arma::mat u, arma::vec d, arma::mat v) {
    if (1 < d.n_cols)
        d = d.diag();

    arma::field<arma::mat> SVD_results(3);
    SVD_results(0) = u;
    SVD_results(1) = d;
    SVD_results(2) = v;

    arma::field<arma::mat> PCA_results = actionet::SVD2PCA(S, SVD_results);

    Rcpp::List res;
    arma::vec s = PCA_results(1).col(0);

    arma::mat X = PCA_results(0);
    for (int i = 0; i < X.n_cols; i++) {
        X.col(i) *= s(i);
    }
    res["x"] = X;
    res["rotation"] = PCA_results(2);
    res["sdev"] = s / std::sqrt(X.n_rows - 1);

    return res;
}

// [[Rcpp::export]]
arma::sp_mat LSI(arma::sp_mat& X, double size_factor = 100000) {
    arma::sp_mat TFIDF = actionet::LSI(X, size_factor);

    return (TFIDF);
}

// [[Rcpp::export]]
arma::mat aggregate_genesets_mahalanobis_2archs(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& marker_mat,
                                                int network_normalization_method = 0,
                                                int expression_normalization_method = 0, int gene_scaling_method = 0,
                                                double pre_alpha = 0.85, double post_alpha = 0.85, int thread_no = 0) {
    arma::mat stats = actionet::aggregate_genesets_mahalanobis_2archs(G, S, marker_mat, network_normalization_method,
                                                                      expression_normalization_method,
                                                                      gene_scaling_method,
                                                                      pre_alpha, post_alpha, thread_no);

    return (stats);
}

// [[Rcpp::export]]
arma::mat aggregate_genesets_mahalanobis_2gmm(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& marker_mat,
                                              int network_normalization_method = 0,
                                              int expression_normalization_method = 0, int gene_scaling_method = 0,
                                              double pre_alpha = 0.85, double post_alpha = 0.85, int thread_no = 0) {
    arma::mat stats = actionet::aggregate_genesets_mahalanobis_2gmm(G, S, marker_mat, network_normalization_method,
                                                                    expression_normalization_method,
                                                                    gene_scaling_method,
                                                                    pre_alpha, post_alpha, thread_no);

    return (stats);
}