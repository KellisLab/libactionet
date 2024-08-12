#include "action_ext.h"

void findConsensus(std::vector<arma::mat> S, full_trace &run_trace, int arch_no, arma::vec alpha,
                   double lambda, int max_it, int) {
    int i;
    int ds_no = run_trace.indiv_trace[arch_no]
            .H_primary.size(); // number of datasets ( ~ 2)
    int cell_no = S[0].n_cols;
    arma::vec c(cell_no);

    // make sure it's a convex vector
    alpha.transform([](double val) { return (std::max(0.0, val)); });
    alpha = arma::normalise(alpha, 1);

    run_trace.indiv_trace[arch_no].C_secondary[0] =
            run_trace.indiv_trace[arch_no].C_primary[0];
    run_trace.indiv_trace[arch_no].H_secondary[0] =
            run_trace.indiv_trace[arch_no].H_primary[0];
    for (int ds = 1; ds < ds_no; ds++) {
        arma::mat G = 1 + cor(trans(run_trace.indiv_trace[arch_no].H_primary[0]),
                              trans(run_trace.indiv_trace[arch_no].H_primary[ds]));

        arma::mat G_matched = MWM_hungarian(G);
        arma::uvec perm = arma::index_max(G_matched, 1);

        run_trace.indiv_trace[arch_no].C_secondary[ds] =
                run_trace.indiv_trace[arch_no].C_primary[ds].cols(perm);
        run_trace.indiv_trace[arch_no].H_secondary[ds] =
                run_trace.indiv_trace[arch_no].H_primary[ds].rows(perm);
    }

    // Compute initial H_consensus
    run_trace.H_consensus[arch_no] =
            zeros(size(run_trace.indiv_trace[arch_no].H_primary[0]));
    for (int ds = 0; ds < ds_no; ds++) {
        run_trace.H_consensus[arch_no] +=
                alpha(ds) * run_trace.indiv_trace[arch_no].H_secondary[ds];
    }

    // Estimate relative ratio of error terms
    arma::mat H_hat = run_trace.H_consensus[arch_no];
    double a = 0.0, b = 0.0, x, y;
    for (int ds = 0; ds < ds_no; ds++) {
        arma::mat W = run_trace.indiv_trace[arch_no].C_secondary[ds];
        arma::mat H = run_trace.indiv_trace[arch_no].H_secondary[ds];

        x = arma::norm(S[ds] - S[ds] * W * H, "fro");
        y = arma::norm(H - H_hat, "fro");
        a += (x * x);
        b += (alpha(ds) * y * y);
    }
    double ratio = a / b;
    lambda *= ratio;

    // Main loop
    for (int it = 0; it < max_it; it++) {
        // Permute rows
        for (int ds = 1; ds < ds_no; ds++) {
            arma::mat G = 1 + arma::cor(arma::trans(run_trace.indiv_trace[arch_no].H_secondary[0]),
                                        arma::trans(run_trace.indiv_trace[arch_no].H_secondary[ds]));
            arma::mat G_matched = MWM_hungarian(G);
            arma::uvec perm = arma::index_max(G_matched, 1);

            run_trace.indiv_trace[arch_no].C_secondary[ds] =
                    run_trace.indiv_trace[arch_no].C_secondary[ds].cols(perm);
            run_trace.indiv_trace[arch_no].H_secondary[ds] =
                    run_trace.indiv_trace[arch_no].H_secondary[ds].rows(perm);
        }

        // Compute shared subspace
        run_trace.H_consensus[arch_no] =
                zeros(size(run_trace.indiv_trace[arch_no].H_primary[0]));
        for (int ds = 0; ds < ds_no; ds++) {
            run_trace.H_consensus[arch_no] +=
                    alpha(ds) * run_trace.indiv_trace[arch_no].H_secondary[ds];
        }

        // Recompute H_i
        for (int ds = 0; ds < ds_no; ds++) {
            arma::mat I = arma::eye(arch_no, arch_no);
            arma::mat Z =
                    S[ds] *
                    run_trace.indiv_trace[arch_no].C_secondary[ds]; // Archetype matrix
            double weight = lambda * alpha[ds];

            arma::mat A = arma::join_vert(arma::trans(Z) * Z, weight * I);
            arma::mat B =
                    arma::join_vert(arma::trans(Z) * S[ds], weight * run_trace.H_consensus[arch_no]);

            run_trace.indiv_trace[arch_no].H_secondary[ds] =
                    ACTIONet::run_simplex_regression(A, B, false);
        }

        // Recompute C_i
        for (int ds = 0; ds < ds_no; ds++) {
            arma::mat W = S[ds] * run_trace.indiv_trace[arch_no].C_secondary[ds];
            arma::mat H = run_trace.indiv_trace[arch_no].H_secondary[ds];
            arma::mat R = S[ds] - W * H;
            for (int j = 0; j < arch_no; j++) {
                double norm_sq = arma::sum(arma::square(H.row(j)));
                arma::vec h = arma::trans(H.row(j)) / norm_sq;
                arma::vec b = R * h + W.col(j);

                c = ACTIONet::run_simplex_regression(S[ds], b, false);

                R += (W.col(j) - S[ds] * c) * H.row(j);
                W.col(j) = S[ds] * c;
                run_trace.indiv_trace[arch_no].C_secondary[ds].col(j) = c;
            }
        }
    }
}

ACTIONet::ACTION_results run_weighted_ACTION(arma::mat &S_r, arma::vec w, int k_min, int k_max, int thread_no,
                                             int max_it, double min_delta) {
    int feature_no = S_r.n_rows;

    stdout_printf("Running weighted ACTION (%d threads):", thread_no);
    FLUSH;

    if (k_max == -1)
        k_max = (int) S_r.n_cols;

    k_min = std::max(k_min, 2);
    k_max = std::min(k_max, (int) S_r.n_cols);

    ACTIONet::ACTION_results trace;

    trace.H = arma::field<arma::mat>(k_max + 1);
    trace.C = arma::field<arma::mat>(k_max + 1);
    trace.selected_cols = arma::field<arma::uvec>(k_max + 1);

    int N = S_r.n_cols;
    if (N != w.n_elem) {
        stderr_printf(
                "Number of elements in the weight vector should match the total number of samples (columns in S_r)\n");
        FLUSH;
        return (trace);
    }

    w = arma::clamp(w, 0, 1);
    arma::mat X_r = arma::normalise(S_r, 1); // ATTENTION!

    arma::mat X_r_scaled = X_r;
    for (int i = 0; i < N; i++) {
        X_r_scaled.col(i) *= w[i];
    }

    int current_k = 0;
    char status_msg[50];

    sprintf(status_msg, "Iterating from k = %d ... %d:", k_min, k_max);
    stderr_printf("\n\t%s %d/%d finished", status_msg, current_k, (k_max - k_min + 1));
    FLUSH;

    mini_thread::parallelFor(k_min, k_max + 1, [&](size_t kk) {
        ACTIONet::SPA_results SPA_res = ACTIONet::run_SPA(X_r_scaled, kk);
        trace.selected_cols[kk] = SPA_res.selected_columns;

        arma::mat W = X_r_scaled.cols(trace.selected_cols[kk]);

        arma::field<arma::mat> AA_res = ACTIONet::run_AA(X_r_scaled, W, max_it, min_delta);

        arma::mat C = AA_res(0);
        arma::mat weighted_archs = X_r_scaled * C;
        arma::mat H = ACTIONet::run_simplex_regression(weighted_archs, X_r, false);
        AA_res(1) = H;

        trace.C[kk] = AA_res(0);
        trace.H[kk] = AA_res(1);
        current_k++;

        stderr_printf("\r\t%s %d/%d finished", status_msg, current_k,
                      (k_max - k_min + 1));
        FLUSH;
    }, thread_no);

    stdout_printf("\r\t%s %d/%d finished\n", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    return trace;
}

Online_ACTION_results run_online_ACTION(arma::mat &S_r, arma::field<arma::uvec> samples,
                                        int k_min, int k_max, int thread_no) {
    int feature_no = S_r.n_rows;

    stdout_printf("Running online ACTION (%d threads):", thread_no);
    FLUSH;

    if (k_max == -1)
        k_max = (int) S_r.n_cols;

    k_min = std::max(k_min, 2);
    k_max = std::min(k_max, (int) S_r.n_cols);

    Online_ACTION_results trace;

    trace.A = arma::field<arma::mat>(k_max + 1);
    trace.B = arma::field<arma::mat>(k_max + 1);
    trace.C = arma::field<arma::mat>(k_max + 1);
    trace.D = arma::field<arma::mat>(k_max + 1);
    trace.selected_cols = arma::field<arma::uvec>(k_max + 1);

    arma::mat X_r_L1 = arma::normalise(S_r, 1, 0);
    arma::mat X_r_L2 = arma::normalise(S_r, 2, 0);

    int current_k = 0;
    char status_msg[50];

    sprintf(status_msg, "Iterating from k = %d ... %d:", k_min, k_max);
    stderr_printf("\n\t%s %d/%d finished", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    mini_thread::parallelFor(k_min, k_max + 1, [&](size_t kk) {
        ACTIONet::SPA_results SPA_res = ACTIONet::run_SPA(X_r_L1, kk);
        trace.selected_cols[kk] = SPA_res.selected_columns;

        arma::mat W = X_r_L2.cols(trace.selected_cols[kk]);

        arma::field<arma::mat> AA_res;
        AA_res = run_online_AA(X_r_L2, W, samples);

        trace.A[kk] = AA_res(0);
        trace.B[kk] = AA_res(1);
        trace.C[kk] = AA_res(2);
        trace.D[kk] = AA_res(3);

        current_k++;

        stderr_printf("\r\t%s %d/%d finished", status_msg, current_k,
                      (k_max - k_min + 1));
        FLUSH;
    }, thread_no);

    stdout_printf("\r\t%s %d/%d finished\n", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    return trace;
}

ACTIONet::ACTION_results
run_ACTION_plus(arma::mat &S_r, int k_min, int k_max, int max_it, double min_delta, int max_trial) {

    stdout_printf("Running ACTION++ (%d threads):");
    FLUSH;

    int D = std::min((int) S_r.n_rows, (int) S_r.n_cols);
    if (k_max == -1)
        k_max = D;

    k_min = std::max(k_min, 2);
    k_max = std::min(k_max, D);

    ACTIONet::ACTION_results trace;

    trace.H = arma::field<arma::mat>(k_max + 1, 1);
    trace.C = arma::field<arma::mat>(k_max + 1, 1);
    trace.selected_cols = arma::field<arma::uvec>(k_max + 1, 1);

    arma::mat X_r = arma::normalise(S_r, 1); // ATTENTION!
    ACTIONet::SPA_results SPA_res = ACTIONet::run_SPA(X_r, D);
    arma::uvec selected_cols = SPA_res.selected_columns;

    arma::mat W = arma::mat(X_r.col(selected_cols(0)));

    arma::field<arma::mat> AA_res;
    int cur_idx = 0, jj, kk;
    stdout_printf("Iterating from k=%d ... %d (max trial = %d)\n", k_min, k_max,
                  max_trial);
    FLUSH;
    for (kk = k_min; kk <= k_max; kk++) {
        stdout_printf("\tk = %d\n", kk);
        FLUSH;

        for (jj = 0; jj < max_trial; jj++) {
            cur_idx++;
            stdout_printf("\t\tTrial %d: candidate %d = %d ... ", jj + 1, cur_idx + 1,
                          selected_cols(cur_idx));
            FLUSH;
            arma::mat W_tmp = arma::join_rows(W, X_r.col(selected_cols(cur_idx)));

            AA_res = ACTIONet::run_AA(X_r, W_tmp, max_it, min_delta);

            arma::vec influential_cells = arma::vec(arma::trans(arma::sum(arma::spones(arma::sp_mat(AA_res(0))), 0)));
            int trivial_counts = (int) sum(influential_cells <= 1);

            if ((trivial_counts == 0)) {
                stdout_printf("success\n");
                FLUSH;
                selected_cols(kk - 1) = selected_cols(cur_idx);
                break;
            }

            stdout_printf("failed\n");
            FLUSH;
            if ((cur_idx == (D - 1))) {
                stdout_printf("Reached end of the line!\n");
                FLUSH;
                break;
            }
        }

        if ((jj == max_trial) || (cur_idx == (D - 1))) {
            break;
        }

        trace.C[kk] = AA_res(0);
        trace.H[kk] = AA_res(1);
        trace.selected_cols(kk) = selected_cols(arma::span(0, kk - 1));

        W = X_r * AA_res(0);
    }

    trace.C = trace.C.rows(0, kk - 1);
    trace.H = trace.H.rows(0, kk - 1);
    trace.selected_cols = trace.selected_cols.rows(0, kk - 1);

    return trace;
}

ACTIONet::ACTION_results run_subACTION(arma::mat &S_r, arma::mat &W_parent, arma::mat &H_parent, int kk, int k_min,
                                       int k_max, int thread_no, int max_it, double min_delta) {
    int feature_no = S_r.n_rows;

    stdout_printf("Running subACTION (%d threads) for parent archetype %d\n",
                  thread_no, kk + 1);
    FLUSH;

    if (k_max == -1)
        k_max = (int) S_r.n_cols;

    k_min = std::max(k_min, 2);
    k_max = std::min(k_max, (int) S_r.n_cols);

    arma::mat X_r = arma::normalise(S_r, 1); // ATTENTION!

    arma::vec h = arma::vec(arma::trans(H_parent.row(kk)));
    arma::mat W_prior = W_parent;
    W_prior.shed_col(kk);

    arma::mat X_r_scaled = X_r; // To deflate or not deflate!

    for (int i = 0; i < X_r_scaled.n_cols; i++) {
        X_r_scaled.col(i) *= h[i];
    }

    ACTIONet::ACTION_results trace;
    trace.H = arma::field<arma::mat>(k_max + 1);
    trace.C = arma::field<arma::mat>(k_max + 1);
    trace.selected_cols = arma::field<arma::uvec>(k_max + 1);

    int current_k = 0;
    char status_msg[50];

    sprintf(status_msg, "Iterating from k = %d ... %d:", k_min, k_max);
    stderr_printf("\n\t%s %d/%d finished", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    stdout_printf("Iterating from k=%d ... %d\n", k_min, k_max);
    FLUSH;

    mini_thread::parallelFor(
            k_min, k_max + 1,
            [&](size_t kkk) {
                ACTIONet::SPA_results SPA_res = ACTIONet::run_SPA(X_r_scaled, kkk);
                trace.selected_cols[kkk] = SPA_res.selected_columns;

                arma::mat W = X_r.cols(trace.selected_cols[kkk]);
                arma::field<arma::mat> AA_res;
                AA_res = run_AA_with_prior(X_r_scaled, W, W_prior, max_it, min_delta);

                trace.C[kkk] = AA_res(0);
                trace.H[kkk] = AA_res(1);
                current_k++;

                stderr_printf("\r\t%s %d/%d finished", status_msg, current_k,
                              (k_max - k_min + 1));
                FLUSH;
            },
            thread_no);

    stdout_printf("\r\t%s %d/%d finished\n", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    return trace;
}

full_trace runACTION_muV(std::vector<arma::mat> S_r, int k_min, int k_max, arma::vec alpha, double lambda, int AA_iters,
                         int Opt_iters, int thread_no) {

    stdout_printf("Running ACTION muV (%d threads):", thread_no);
    FLUSH;

    double lambda2 = 1e-5, epsilon = 1e-5;

    k_min = std::max(k_min, 2);
    k_max = std::min(k_max, (int) (S_r[0].n_cols));

    int cell_no = S_r[0].n_cols;
    arma::vec c(cell_no);

    full_trace run_trace;
    run_trace.H_consensus.resize(k_max + 1);
    run_trace.indiv_trace.resize(k_max + 1);
    for (int kk = 0; kk <= k_max; kk++) {
        run_trace.indiv_trace[kk].selected_cols.resize(S_r.size());
        run_trace.indiv_trace[kk].H_primary.resize(S_r.size());
        run_trace.indiv_trace[kk].C_primary.resize(S_r.size());
        run_trace.indiv_trace[kk].H_secondary.resize(S_r.size());
        run_trace.indiv_trace[kk].C_secondary.resize(S_r.size());
        run_trace.indiv_trace[kk].C_consensus.resize(S_r.size());
    }

    // Normalize signature profiles
    for (int i = 0; i < S_r.size(); i++) {
        S_r[i] = arma::normalise(S_r[i], 1, 0); // norm-1 normalize across columns --
        // particularly important for SPA
    }

    arma::field<arma::mat> AA_res(2, 1);
    int current_k = 0;
    char status_msg[50];

    sprintf(status_msg, "Iterating from k = %d ... %d:", k_min, k_max);
    stderr_printf("\n\t%s %d/%d finished", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    mini_thread::parallelFor(
            k_min, k_max + 1,
            [&](size_t kk) {
                // Solve ACTION for a fixed-k to "jump-start" the joint optimization
                // problem.
                for (int i = 0; i < S_r.size(); i++) {
                    ACTIONet::SPA_results SPA_res = ACTIONet::run_SPA(S_r[i], kk);
                    run_trace.indiv_trace[kk].selected_cols[i] = SPA_res.selected_columns;

                    arma::mat W = S_r[i].cols(run_trace.indiv_trace[kk].selected_cols[i]);

                    AA_res = ACTIONet::run_AA(S_r[i], W, AA_iters, 1e-16);

                    arma::mat C0 = AA_res(0);
                    C0.transform([](double val) { return (std::min(1.0, std::max(0.0, val))); });
                    C0 = arma::normalise(C0, 1);
                    run_trace.indiv_trace[kk].C_primary[i] = C0;

                    arma::mat H0 = AA_res(1);
                    H0.transform([](double val) { return (std::min(1.0, std::max(0.0, val))); });
                    H0 = arma::normalise(H0, 1);
                    run_trace.indiv_trace[kk].H_primary[i] = H0;
                }

                // Compute consensus latent subspace, H^*
                findConsensus(S_r, run_trace, kk, alpha, lambda, Opt_iters,
                              thread_no); // sets secondary and consensus objects

                // decouple to find individual consensus C matrices
                for (int i = 0; i < S_r.size(); i++) {
                    arma::mat S = S_r[i];
                    arma::mat C = run_trace.indiv_trace[kk].C_secondary[i];
                    arma::mat H = run_trace.indiv_trace[kk].H_secondary[i];
                    arma::mat W = S * C;

                    arma::mat R = S - W * H;

                    run_trace.indiv_trace[kk].C_consensus[i] = arma::zeros(cell_no, kk);
                    for (int j = 0; j < kk; j++) {
                        arma::vec w = W.col(j);
                        arma::vec h = arma::trans(H.row(j));

                        double norm_sq = arma::dot(h, h);
                        if (norm_sq < double(10e-8)) {
                            // singular
                            int max_res_idx = arma::index_max(arma::rowvec(arma::sum(arma::square(R), 0)));
                            W.col(j) = S.col(max_res_idx);
                            c.zeros();
                            c(max_res_idx) = 1;
                            C.col(j) = c;
                        } else {
                            arma::vec b = w;
                            cblas_dgemv(CblasColMajor, CblasNoTrans, R.n_rows, R.n_cols,
                                        (1.0 / norm_sq), R.memptr(), R.n_rows, h.memptr(), 1,
                                        1, b.memptr(), 1);

                            C.col(j) = ACTIONet::run_simplex_regression(S, b, false);

                            arma::vec w_new = S * C.col(j);
                            arma::vec delta = (w - w_new);

                            // Rank-1 update: R += delta*h
                            cblas_dger(CblasColMajor, R.n_rows, R.n_cols, 1.0, delta.memptr(),
                                       1, h.memptr(), 1, R.memptr(), R.n_rows);

                            W.col(j) = w_new;
                        }

                        run_trace.indiv_trace[kk].C_consensus[i].col(j) = C.col(j);
                    }
                }
                current_k++;
                stderr_printf("\r\t%s %d/%d finished", status_msg, current_k,
                              (k_max - k_min + 1));
                FLUSH;
            },
            thread_no);

    stdout_printf("\r\t%s %d/%d finished\n", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    return run_trace;
}
