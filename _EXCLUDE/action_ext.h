#ifndef ACTION_EXT_H
#define ACTION_EXT_H

#include "action.hpp"
#include "aa_ext.h"
#include "tools/mwm.hpp"

// Structs
// To store the output of run_online_ACTION()
struct Online_ACTION_results {
    arma::field<arma::uvec> selected_cols;
    arma::field<arma::mat> A;
    arma::field<arma::mat> B;
    arma::field<arma::mat> C;
    arma::field<arma::mat> D;
};

struct mvtrace_obj {
    std::vector<arma::uvec> selected_cols;

    std::vector<arma::mat> H_primary;
    std::vector<arma::mat> C_primary;

    std::vector<arma::mat> H_secondary;
    std::vector<arma::mat> C_secondary;

    std::vector<arma::mat> C_consensus;
};

struct full_trace {
    std::vector<mvtrace_obj> indiv_trace;
    std::vector<arma::mat> H_consensus;
};

// Functions
ACTIONet::ACTION_results run_weighted_ACTION(arma::mat &S_r, arma::vec w, int k_min, int k_max,
                                             int thread_no, int max_it = 50,
                                             double min_delta = 1e-16);

Online_ACTION_results run_online_ACTION(arma::mat &S_r, arma::field<arma::uvec> samples,
                                        int k_min, int k_max, int thread_no);

ACTIONet::ACTION_results run_ACTION_plus(arma::mat &S_r, int k_min, int k_max, int max_it = 100,
                                         double min_delta = 1e-16, int max_trial = 3);

ACTIONet::ACTION_results run_subACTION(arma::mat &S_r, arma::mat &W_parent, arma::mat &H_parent, int kk,
                                       int k_min, int k_max, int thread_no,
                                       int max_it = 50, double min_delta = 1e-16);

full_trace runACTION_muV(std::vector<arma::mat> S_r, int k_min, int k_max, arma::vec alpha,
                         double lambda, int AA_iters, int Opt_iters,
                         int thread_no);

#endif //ACTION_EXT_H
