#include "aa_ext.h"

// Solves the weighted Archetypal Analysis (AA) problem
arma::field<arma::mat> run_weighted_AA(arma::mat &A, arma::mat &W0, arma::vec w, int max_it, double min_delta) {
    int N = A.n_cols;
    arma::field<arma::mat> decomposition(2);

    if (N != w.n_elem) {
        stdout_printf(
                "Number of elements in the weight vector should match the total number of samples (columns in A)\n");
        FLUSH;
        return (decomposition);
    }

    w = arma::clamp(w, 0, 1);
    arma::mat A_scaled = A;
    for (int i = 0; i < N; i++) {
        A_scaled.col(i) *= w[i];
    }
    decomposition = actionet::run_AA(A_scaled, W0, max_it, min_delta);

    arma::mat C = decomposition(0);
    arma::mat weighted_archs = A_scaled * C;
    arma::mat H = actionet::run_simplex_regression(weighted_archs, A, false);
    decomposition(1) = H;

    return (decomposition);
}

arma::field<arma::mat> Online_update_AA(arma::mat &Xt, arma::mat &D, arma::mat &A, arma::mat &B) {
    // Compute archetype coefficients using the last learned dictionary
    arma::mat Ct = actionet::run_simplex_regression(D, Xt, false);

    // Just in case!
    Ct = arma::clamp(Ct, 0, 1);
    Ct = arma::normalise(Ct, 1);
    arma::mat Ct_T = arma::trans(Ct);

    // Update sufficient statistics
    arma::mat At = A + Ct * Ct_T;
    arma::mat Bt = B + Xt * Ct_T;

    // Update the dictionary using block-coordinate-descent (BCD)
    arma::mat Dt(arma::size(D));
    for (int j = 0; j < D.n_cols; j++) {
        arma::vec u = D.col(j) + (1.0 / At(j, j)) * (Bt.col(j) - D * At.col(j));
        Dt.col(j) = u / std::max(arma::norm(u, 2), 1.0);
    }

    arma::field<arma::mat> decomposition(4, 1);

    decomposition(0) = At;
    decomposition(1) = Bt;
    decomposition(2) = Ct;
    decomposition(3) = Dt;

    return (decomposition);
}

arma::field<arma::mat> run_online_AA(arma::mat &X, arma::mat &D0, arma::field<arma::uvec> samples) {
    int m = X.n_rows;
    int n = X.n_cols;
    int k = D0.n_cols;

    arma::mat At = arma::zeros(k, k);
    arma::mat Bt = arma::zeros(m, k);
    arma::mat Ct(k, n), Ct_T(n, k);

    // Just in case
    X = arma::normalise(X, 2, 0);
    arma::mat Dt = arma::normalise(D0, 2, 0);

    for (int t = 0; t < samples.n_elem; t++) {
        // Extract the next batch
        arma::uvec idx = samples(t);
        arma::mat Xt = X.cols(idx);

        // Compute archetype coefficients using the last learned dictionary
        Ct = actionet::run_simplex_regression(Dt, Xt, false);
        Ct_T = arma::trans(Ct);

        // Update sufficient statistics
        At += Ct * Ct_T;
        Bt += Xt * Ct_T;

        // Update the dictionary using block-coordinate-descent (BCD)
        for (int j = 0; j < k; j++) {
            arma::vec u = Dt.col(j) + (1.0 / At(j, j)) * (Bt.col(j) - Dt * At.col(j));
            Dt.col(j) = u / std::max(norm(u, 2), 1.0);
        }
    }

    // Just in case!
    Ct = arma::clamp(Ct, 0, 1);
    Ct = arma::normalise(Ct, 1);

    arma::field<arma::mat> decomposition(4, 1);

    decomposition(0) = At;
    decomposition(1) = Bt;
    decomposition(2) = Ct;
    decomposition(3) = Dt;

    return (decomposition);
}

arma::field<arma::mat>
run_AA_with_prior(arma::mat &A, arma::mat &W0, arma::mat &W_prior, int max_it, double min_delta) {
    int sample_no = A.n_cols;
    int d = A.n_rows;  // input dimension
    int k = W0.n_cols; // AA components

    arma::mat C = arma::zeros(sample_no, k);
    arma::mat H = arma::zeros(k, sample_no);

    arma::mat W = W0;
    arma::vec c(sample_no);

    for (int it = 0; it < max_it; it++) {
        arma::mat combined_W = join_rows(W, W_prior);
        arma::mat combined_H = actionet::run_simplex_regression(combined_W, A, true);

        H = combined_H.rows(arma::span(0, k - 1));

        arma::mat R = A - W * H;
        arma::mat Ht = trans(H);
        for (int i = 0; i < k; i++) {
            arma::vec w = W.col(i);
            arma::vec h = Ht.col(i);

            double norm_sq = arma::dot(h, h);
            if (norm_sq < double(10e-8)) {
                // singular
                int max_res_idx = arma::index_max(arma::rowvec(sum(square(R), 0)));
                W.col(i) = A.col(max_res_idx);
                c.zeros();
                c(max_res_idx) = 1;
                C.col(i) = c;
            } else {
                arma::vec b = w;
                cblas_dgemv(CblasColMajor, CblasNoTrans, R.n_rows, R.n_cols,
                            (1.0 / norm_sq), R.memptr(), R.n_rows, h.memptr(), 1, 1,
                            b.memptr(), 1);

                // No matching signature for `IRLB_SVD`. Likely old or bug.
                C.col(i) = actionet::IRLB_SVD(A, b, false);

                arma::vec w_new = A * C.col(i);
                arma::vec delta = (w - w_new);

                // Rank-1 update: R += delta*h
                cblas_dger(CblasColMajor, R.n_rows, R.n_cols, 1.0, delta.memptr(), 1,
                           h.memptr(), 1, R.memptr(), R.n_rows);

                W.col(i) = w_new;
            }
        }
    }

    C = arma::clamp(C, 0, 1);
    C = arma::normalise(C, 1);
    H = arma::clamp(H, 0, 1);
    H = arma::normalise(H, 1);

    arma::field<arma::mat> decomposition(2, 1);
    decomposition(0) = C;
    decomposition(1) = H;

    return decomposition;
}

Coreset compute_AA_coreset(arma::sp_mat &S, int m) {
    arma::vec mu = arma::vec(arma::mean(S, 1));

    arma::vec q = arma::zeros(S.n_cols);
    arma::sp_mat::const_iterator it = S.begin();
    arma::sp_mat::const_iterator it_end = S.end();
    for (; it != it_end; ++it) {
        double d = (*it) - mu[it.row()];
        q[it.col()] += d * d;
    }
    arma::vec p = q / arma::sum(q); // sampling probability

    if (m == 0) {
        double p_sum = arma::sum(p);
        double p_sq_sum = arma::sum(arma::square(p));
        m = (int) ((p_sum * p_sum) / (p_sq_sum));
    }
    m = std::min(m, (int) S.n_cols);

    arma::vec p_sorted = arma::sort(p, "descend");
    double p_threshold = p_sorted(m - 1);

    arma::uvec sample_idx = arma::find(p_threshold <= p);
    m = sample_idx.n_elem; // Can do actual sampling, maybe!

    arma::mat S_coreset(S.n_rows, m);
    for (int j = 0; j < m; j++) {
        S_coreset.col(j) = arma::vec(S.col(sample_idx(j)));
    }
    arma::vec w_coreset = 1.0 / (m * p(sample_idx));

    Coreset coreset;
    coreset.S_coreset = S_coreset;
    coreset.w_coreset = w_coreset;
    coreset.index = sample_idx;

    return (coreset);
}
