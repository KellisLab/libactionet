#include "action_post_ext.h"

double Kappa(double p, double q) {
    double a = 0.0, b = 0.0;
    if ((1e-300 < p) & (1e-300 < q)) {
        a = p * std::log(p / q);
    }
    if ((p < (1 - 1e-300)) & (q < (1 - 1e-300))) {
        b = (1 - p) * std::log((1 - p) / (1 - q));
    }

    double k = a + b;
    return (k);
}

double log_HGT_tail(int population_size, int success_count, int sample_size, int observed_success) {

    if (observed_success == 0)
        return (0);

    double success_rate = (double) success_count / population_size;
    double expected_success = sample_size * success_rate;
    double delta = (observed_success / expected_success) - 1.0;
    if (delta < 0) {
        return (0);
    }

    double log_tail_bound =
            sample_size * Kappa((1.0 + delta) * success_rate, success_rate);

    return (log_tail_bound);
}

double assess_overlap(arma::uvec i1, arma::uvec i2, int population_size) {

    int success_count = i1.n_elem;
    int sample_size = i2.n_elem;

    arma::uvec shared = arma::intersect(i1, i2);
    int observed_success = shared.n_elem;

    double log_pval = log_HGT_tail(population_size, success_count, sample_size,
                                   observed_success);

    return (log_pval);
}

arma::mat compute_overlap_matrix(arma::mat C) {

    int N = C.n_cols;
    arma::mat O = arma::zeros(N, N);

    std::vector<arma::uvec> indices(N);
    for (int i = 0; i < N; i++) {
        arma::uvec idx = arma::find(C.col(i) > 0);
        indices[i] = idx;
    }

    for (int i = 0; i < N; i++) {
        arma::uvec i1 = indices[i];
        for (int j = i + 1; j < N; j++) {
            arma::uvec i2 = indices[j];

            O(i, j) = O(j, i) = assess_overlap(i1, i2, N);
        }
    }

    return (O);
}

arma::mat NetEnh(arma::mat A) {
    A.diag().zeros();
    arma::mat P = arma::normalise(A, 1, 1);

    arma::mat D = arma::diagmat(1.0 / (arma::sqrt(arma::sum(P) + 1e-16)));
    arma::mat W = P * D;
    arma::mat P2 = W * arma::trans(W);

    return (P2);
}

arma::field<arma::mat> nndsvd(arma::mat &A, int dim, int max_iter) {
    dim = std::min(dim, (int) A.n_cols);
    arma::field<arma::mat> SVD_res = actionet::svdHalko(A, dim, max_iter, 0, 0);

    arma::mat U = SVD_res(0);
    arma::vec s = SVD_res(1);
    arma::mat V = SVD_res(2);

    arma::mat W(arma::size(U));
    arma::mat H(arma::size(V));

    arma::uvec mask_idx;
    for (int i = 0; i < U.n_cols; i++) {
        arma::vec u = U.col(i);
        arma::vec v = V.col(i);

        arma::vec up = u;
        mask_idx = arma::find(u < 0);
        if (mask_idx.n_elem > 0)
            up(mask_idx).zeros();

        arma::vec un = -u;
        mask_idx = arma::find(u > 0);
        if (mask_idx.n_elem > 0)
            un(mask_idx).zeros();

        arma::vec vp = v;
        mask_idx = arma::find(v < 0);
        if (mask_idx.n_elem > 0)
            vp(mask_idx).zeros();

        arma::vec vn = -v;
        mask_idx = arma::find(v > 0);
        if (mask_idx.n_elem > 0)
            vn(mask_idx).zeros();

        double n_up = arma::norm(up);
        double n_un = arma::norm(un);
        double n_vp = arma::norm(vp);
        double n_vn = arma::norm(vn);

        double termp = n_up * n_vp;
        double termn = n_un * n_vn;
        if (termp >= termn) {
            W.col(i) = sqrt(s(i) * termp) * up / n_up;
            H.col(i) = sqrt(s(i) * termp) * vp / n_vp;
        } else {
            W.col(i) = sqrt(s(i) * termn) * un / n_un;
            H.col(i) = sqrt(s(i) * termn) * vn / n_vn;
        }
    }

    arma::field<arma::mat> out(5);
    out(0) = W;
    out(1) = H;
    out(2) = s;
    out(3) = U;
    out(4) = V;

    return (out);
}

arma::field<arma::mat> orient_SVD(arma::field<arma::mat> &SVD_out) {
    arma::mat U = SVD_out(0);
    arma::vec s = SVD_out(1);
    arma::mat V = SVD_out(2);

    int dim = U.n_cols;

    arma::mat Up = U;
    arma::mat Vp = V;

    arma::uvec mask_idx;
    for (int i = 0; i < U.n_cols; i++) {
        arma::vec u = U.col(i);
        arma::vec v = V.col(i);

        arma::vec up = u;
        mask_idx = arma::find(u < 0);
        if (mask_idx.n_elem > 0)
            up(mask_idx).zeros();

        arma::vec un = -u;
        mask_idx = arma::find(u > 0);
        if (mask_idx.n_elem > 0)
            un(mask_idx).zeros();

        arma::vec vp = v;
        mask_idx = arma::find(v < 0);
        if (mask_idx.n_elem > 0)
            vp(mask_idx).zeros();

        arma::vec vn = -v;
        mask_idx = arma::find(v > 0);
        if (mask_idx.n_elem > 0)
            vn(mask_idx).zeros();

        double n_up = arma::norm(up);
        double n_un = arma::norm(un);
        double n_vp = arma::norm(vp);
        double n_vn = arma::norm(vn);

        double termp = n_up * n_vp;
        double termn = n_un * n_vn;
        if (termp < termn) {
            Up.col(i) *= -1;
            Vp.col(i) *= -1;
        }
    }

    arma::field<arma::mat> out(3);
    out(0) = Up;
    out(1) = s;
    out(2) = Vp;

    return (out);
}

arma::field<arma::mat> convexSVD(arma::mat &A, int dim, int max_iter) {
    arma::field<arma::mat> out(4);

    dim = std::min(dim, (int) A.n_cols);
    arma::field<arma::mat> SVD_res = actionet::svdHalko(A, dim, max_iter, 0, 0);
    SVD_res = orient_SVD(SVD_res);

    arma::mat U = SVD_res(0);
    arma::vec s = SVD_res(1);
    arma::mat V = SVD_res(2);

    out(0) = U;
    out(1) = s;
    out(2) = V;

    return (out);
}

arma::field<arma::mat> recursiveNMU(arma::mat M, int dim, int max_SVD_iter, int max_iter_inner) {
    dim = std::min(dim, (int) M.n_cols);

    arma::mat W(M.n_rows, dim);
    arma::mat H(M.n_cols, dim);
    arma::vec obj(dim);
    arma::vec factor_weights(dim);
    arma::vec ss(dim);

    // sparse M_sp = sparse(M);

    double denom = arma::sum(arma::sum(arma::square(M)));
    for (int k = 0; k < dim; k++) {
        arma::field<arma::mat> SVD_res = actionet::svdHalko(M, 1, max_SVD_iter, 0, 0);
        arma::mat U = SVD_res(0);
        arma::vec s = SVD_res(1);
        arma::mat V = SVD_res(2);

        arma::vec x = arma::abs(U.col(0)) * sqrt(s(0));
        arma::vec y = arma::abs(V.col(0)) * sqrt(s(0));

        W.col(k) = x;
        H.col(k) = y;

        arma::mat R = M - x * arma::trans(y);
        arma::mat lambda = -R;
        lambda.transform([](double val) { return (val < 0 ? 0 : val); });

        for (int j = 0; j < max_iter_inner; j++) {
            arma::mat A = M - lambda;

            x = A * y;
            x.transform([](double val) { return (val < 0 ? 0 : val); });
            x /= (arma::max(x) + 1e-16);

            y = arma::trans(arma::trans(x) * A);
            y.transform([](double val) { return (val < 0 ? 0 : val); });
            y /= arma::dot(x, x);

            if ((arma::sum(x) != 0) && (arma::sum(y) != 0)) {
                W.col(k) = x;
                H.col(k) = y;
                R = M - x * arma::trans(y);
                lambda = lambda - R / ((double) j + 1);
                lambda.transform([](double val) { return (val < 0 ? 0 : val); });
            } else {
                lambda /= 2.0;
                x = W.col(k);
                y = H.col(k);
            }
        }

        arma::mat oldM = M;
        M -= x * arma::trans(y);
        M.transform([](double val) { return (val < 0 ? 0 : val); });

        obj(k) = (arma::sum(arma::sum(arma::square(M))) / denom);
        ss(k) = s(0);

        double w_norm1 = arma::sum(arma::abs(W.col(k))); // abs() is reducndant
        W.col(k) /= w_norm1;
        double h_norm1 = arma::sum(arma::abs(H.col(k))); // abs() is reducndant
        H.col(k) /= h_norm1;

        factor_weights(k) = w_norm1 * h_norm1;
    }

    arma::field<arma::mat> out(5);
    out(0) = W;
    out(1) = H;
    out(2) = factor_weights;
    out(3) = obj;
    out(4) = ss;

    return (out);
}

arma::field<arma::mat> recursiveNMU_mine(arma::mat M, int dim, int max_SVD_iter, int max_iter_inner) {
    dim = std::min(dim, (int) M.n_cols);

    arma::mat W(M.n_rows, dim);
    arma::mat H(M.n_cols, dim);
    arma::vec factor_weights(dim);

    arma::mat M0 = M;

    arma::vec s;
    arma::mat U, V;
    double denom = arma::sum(arma::sum(arma::square(M)));
    for (int k = 0; k < dim; k++) {
        arma::field<arma::mat> SVD_res = actionet::IRLB_SVD(M, 1, max_SVD_iter, 0, 0);
        U = SVD_res(0);
        s = SVD_res(1);
        V = SVD_res(2);

        arma::vec w = arma::trans(arma::trans(U.col(0)) * M);
        int selected_columns = arma::index_max(w);

        arma::vec u = M0.col(selected_columns);
        W.col(k) = u;
        H.col(k).zeros();
        H(selected_columns, k) = 1;

        arma::vec v = M.col(selected_columns);
        M -= v * (arma::trans(v) * M) / arma::dot(v, v);
        M.transform([](double val) { return (val < 0 ? 0 : val); });

        factor_weights(k) = (arma::sum(arma::sum(arma::abs(M))) / M.n_cols);
    }

    arma::field<arma::mat> out(3);
    out(0) = W;
    out(1) = H;
    out(2) = factor_weights;

    return (out);
}

arma::vec sweepcut(arma::sp_mat &A, arma::vec s, int min_size, int max_size) {
    A.diag().zeros();
    int nV = A.n_rows;
    if (max_size == -1) {
        max_size = nV;
    }

    arma::vec w = arma::vec(arma::sum(A, 1));
    double total_vol = arma::sum(w);

    arma::vec conductance = arma::datum::inf * arma::ones(max_size);

    arma::uvec perm = arma::sort_index(s, "descend");
    arma::vec x = arma::zeros(nV);
    x(perm(arma::span(0, min_size - 1))).ones();
    double vol = arma::sum(w(perm(arma::span(0, min_size - 1))));

    double cut_size = vol;
    for (int i = 0; i < min_size; i++) {
        for (int j = 0; j < min_size; j++) {
            cut_size -= A(i, j);
        }
    }

    for (int i = min_size; i <= max_size - 1; i++) {
        int u = perm(i);
        vol += w[u];

        x(u) = 1;

        arma::sp_mat::col_iterator it = A.begin_col(u);
        for (; it != A.end_col(u); it++) {
            int v = it.row();
            if (x[v] == 0) { // v is in S_prime (not yet selected)
                cut_size += (*it);
            } else {
                cut_size -= (*it);
            }
        }

        double vol_prime = total_vol - vol;
        conductance(i) = cut_size / std::min(vol, vol_prime);
    }

    return (conductance);
}
