// [[Rcpp::interfaces(r, cpp)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include "libactionet_config.hpp"
#include "action.hpp"
#include "actionet.hpp"
#include "visualization.hpp"
#include "tools.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List
run_ACTION_muV(const Rcpp::List &S, int k_min, int k_max, arma::vec alpha, double lambda = 1, int AA_iters = 50,
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
//' @return A named list with entries 'selected_columns' and 'norms'
//' @examples
//' H = run_SPA(S_r, 10)
// [[Rcpp::export]]
Rcpp::List run_SPA_rows_sparse(arma::sp_mat &A, int k) {
    ACTIONet::SPA_results res = run_SPA_rows_sparse(A, k);
    arma::uvec selected_columns = res.selected_columns;

    arma::vec cols(k);
    for (int i = 0; i < k; i++) {
        cols[i] = selected_columns[i] + 1;
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
//' values of k ' @examples ' ACTION.out = run_ACTION_plus(S_r, k_max = 10) ' H8
//' = ACTION.out$H[[8]] ' cell.assignments = apply(H8, 2, which.max)
// [[Rcpp::export]]
Rcpp::List run_ACTION_plus(arma::mat &S_r, int k_min = 2, int k_max = 30, int max_it = 100, double min_delta = 1e-6,
                           int max_trial = 3) {

    ACTION_results trace = run_ACTION_plus(S_r, k_min, k_max, max_it, min_delta, max_trial);

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
// [[Rcpp::export]]
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
// [[Rcpp::export]]
Rcpp::List run_weighted_ACTION(arma::mat &S_r, arma::vec w, int k_min = 2, int k_max = 30, int thread_no = 0,
                               int max_it = 50, double min_delta = 1e-16) {

    ACTION_results trace = run_weighted_ACTION(S_r, w, k_min, k_max, thread_no, max_it, min_delta);

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