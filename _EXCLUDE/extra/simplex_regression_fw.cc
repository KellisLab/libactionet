#include "simplex_regression_fw.h"

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define now() std::chrono::high_resolution_clock::now()

// Re-implemented from: Fast and Robust Archetypal Analysis for Representation
// Learning

// min(|| AX - B ||) s.t. simplex constraint
arma::mat runSimplexRegression_FW_base(arma::mat &A, arma::mat &B, int max_iter, double min_diff) {

    if (max_iter == -1)
        max_iter = A.n_cols;

    stdout_printf("Initializing ... ");
    // mat tmp = trans(A) * B;
    arma::mat tmp = cor(A, B);

    arma::mat X = arma::zeros(arma::size(tmp));

    for (int j = 0; j < X.n_cols; j++) {
        arma::uword i = arma::index_max(tmp.col(j));
        X(i, j) = 1;
        // X(0, j) = 1;
    }
    stdout_printf("done\n");

    arma::mat At = arma::trans(A);
    arma::mat AtA = At * A;
    arma::mat AtB = At * B;

    arma::mat old_X = X;
    for (int it = 0; it < max_iter; it++) {
        arma::mat grad = (AtA * X) - AtB;
        arma::mat obj = A * X - B;

        for (int k = 0; k < X.n_cols; k++) {
            arma::vec g = grad.col(k);
            arma::vec x = X.col(k);

            int i1, i2;
            int i1_val, i2_val;
            i1_val = i2_val = 1000;
            i1 = i2 = 0;
            for (int i = 0; i < g.n_elem; i++) {
                if (g(i) < i1_val) {
                    i1_val = g(i);
                    i1 = i;
                }
                if (x(i) > 0) {
                    if (g(i) < i2_val) {
                        i2_val = g(i);
                        i2 = i;
                    }
                }
            }

            arma::vec d_FW = -x;
            d_FW(i1) = 1 + d_FW(i1);

            arma::vec d_A = x;
            d_A(i2) = d_A(i2) - 1;

            double alpha_max = 1;
            arma::vec d;
            if (arma::dot(g, d_FW) < arma::dot(g, d_A)) {
                d = d_FW;
                alpha_max = 1;
            } else {
                d = d_A;
                alpha_max = x(i1) / (1 - x(i1));
            }

            /*
            // Backtracking line-search
            vec Ad = A * d;
            double e1 = dot(Ad, Ad);
            double alpha = 0;
            if(e1 != 0) {
                double e2 = 2 * dot(obj.col(k), Ad);
                double e3 = 0.5* dot(g, d); // multiplier can be in (0, 0.5]
                alpha = (e3 - e2) / e1;
            }
            */
            double alpha = 2.0 / (it + 2);

            X.col(k) = x + alpha * d;
        }

        stdout_printf("%d- ", it);
        double res = arma::sum(arma::sum(arma::abs(old_X - X))) / X.n_cols;
        stdout_printf("%e\n", res);

        if (res < min_diff) {
            break;
        }
        old_X = X;
    }

    X = arma::clamp(X, 0, 1);
    X = arma::normalise(X, 1);

    return (X);
}

arma::mat runSimplexRegression_FW_test1(arma::mat &A, arma::mat &B, int max_iter, double min_diff) {
    if (max_iter == -1)
        max_iter = A.n_cols;

    arma::mat X = arma::zeros(A.n_cols, B.n_cols);
    X.row(0).ones();

    arma::mat At = arma::trans(A);
    arma::mat AtA = At * A;

    arma::vec d;
    arma::mat old_X = X;
    for (int it = 0; it < max_iter; it++) {
        arma::mat grad = (AtA * X) - At * B;
        arma::mat obj = A * X - B;

        arma::mat mask = X;
        mask.transform([](double val) { return (val == 0 ? arma::datum::inf : 1.0); });

        arma::mat masked_grad = grad % mask;
        arma::urowvec ii1 = arma::index_min(grad);
        arma::urowvec ii2 = arma::index_min(masked_grad);

        arma::mat D_FW = -X;
        arma::mat D_A = X;
        arma::vec alpha_caps(X.n_cols);
        for (int j = 0; j < X.n_cols; j++) {
            double x = X(ii1(j), j);
            alpha_caps(j) = x / (1 - x);
            D_FW(ii1(j), j)
                    ++;
            D_A(ii2(j), j)
                    --;
        }

        for (int k = 0; k < X.n_cols; k++) {
            arma::vec g = grad.col(k);
            arma::vec x = X.col(k);

            arma::vec d_FW = D_FW.col(k);
            arma::vec d_A = D_A.col(k);

            double alpha_max = 1;
            if (arma::dot(g, d_FW) < arma::dot(g, d_A)) {
                d = d_FW;
            } else {
                d = d_A;
                alpha_max = alpha_caps(k);
            }

            // Backtracking line-search
            arma::vec Ad = A * d;
            double e1 = arma::dot(Ad, Ad);
            double alpha = 0;
            if (e1 != 0) {
                double e2 = 2 * arma::dot(obj.col(k), Ad);
                double e3 = 0.5 * arma::dot(g, d); // multiplier can be in (0, 0.5]
                alpha = (e3 - e2) / e1;
            }

            // double alpha = 2.0 / (it+ 2);
            alpha = std::min(alpha, alpha_max);

            X.col(k) = x + alpha * d;
        }

        stdout_printf("%d- ", it);
        old_X = X;
    }

    return (X);
}

arma::mat runSimplexRegression_FW_working(arma::mat &A, arma::mat &B, int max_iter, double min_diff) {

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;
    std::chrono::duration<double> elapsed;
    std::chrono::high_resolution_clock::time_point start, finish;

    if (max_iter == -1)
        max_iter = A.n_cols;

    start = now();

    arma::mat tmp = arma::cor(A, B);
    arma::mat X = arma::zeros(arma::size(tmp));

    for (int j = 0; j < X.n_cols; j++) {
        arma::vec v = tmp.col(j);
        int i = arma::index_max(v);
        X(i, j) = 1;
    }

    arma::mat At = arma::trans(A);
    arma::mat AtA = At * A;
    arma::mat AtB = At * B;
    finish = now();
    t1 += duration(finish - start);

    arma::mat old_X = X;
    for (int it = 0; it < max_iter; it++) {

        start = now();
        arma::mat grad = (AtA * X) - AtB;
        arma::mat AX = A * X;
        arma::mat obj = AX - B;
        finish = now();
        t2 += duration(finish - start);

        for (int k = 0; k < X.n_cols; k++) {

            start = now();
            arma::vec g = grad.col(k);
            arma::vec x = X.col(k);
            arma::vec b = B.col(k);

            int i1, i2;
            int i1_val, i2_val;
            i1_val = i2_val = 1000;
            i1 = i2 = 0;
            for (int i = 0; i < g.n_elem; i++) {
                if (g(i) < i1_val) {
                    i1_val = g(i);
                    i1 = i;
                }
                if (x(i) > 0) {
                    if (g(i) < i2_val) {
                        i2_val = g(i);
                        i2 = i;
                    }
                }
            }
            finish = now();
            t3 += duration(finish - start);

            start = now();
            arma::vec d_FW = -x;
            d_FW(i1) = 1 + d_FW(i1);

            arma::vec d_A = x;
            d_A(i2) = d_A(i2) - 1;

            double alpha_max = 1;
            arma::vec d;
            int direction = 0;
            if (arma::dot(g, d_FW) < arma::dot(g, d_A)) {
                direction = +1; // Adding element
                d = d_FW;
                alpha_max = 1;
            } else {
                direction = -1; // Removing element
                d = d_A;
                alpha_max = x(i2) / (1 - x(i2));
            }
            finish = now();
            t4 += duration(finish - start);

            start = now();

            arma::vec q;
            if (direction == +1) {
                q = A.unsafe_col(i1) - AX.unsafe_col(k);
            } else {
                q = AX.unsafe_col(k) - A.unsafe_col(i2);
            }
            // vec q = A * d;

            /*
                        double alpha = 0;
                        double q_norm = norm(q, 1);
                        if(q_norm > 0) {
                            double q_norm_sq = q_norm*q_norm;
                            vec delta = -obj.col(k);
                            alpha = dot(q, delta) / q_norm_sq;
                        }
            */
            double alpha = 2 / (it + 2);

            alpha = std::min(alpha, alpha_max);
            // X.col(k) = x + alpha*d;
            finish = now();
            t5 += duration(finish - start);
        }

        start = now();
        //("%d- ", it);
        double res = arma::norm(old_X - X, "fro");

        finish = now();
        t6 += duration(finish - start);

        old_X = X;
    }

    double total = t1 + t2 + t3 + t4 + t5 + t6;

    X = arma::clamp(X, 0, 1);
    X = arma::normalise(X, 1);

    return (X);
}

arma::mat runSimplexRegression_FW(arma::mat &A, arma::mat &B, int max_iter, double min_diff) {

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;
    std::chrono::duration<double> elapsed;
    std::chrono::high_resolution_clock::time_point start, finish;

    int m = A.n_rows, n = A.n_cols, k = B.n_cols;
    bool compute_AtA = n < 1000; //(n*n*(m+k)) < (2*m*n*k);

    if (max_iter == -1)
        max_iter = A.n_cols;

    start = now();
    arma::mat tmp = arma::cor(A, B);
    arma::mat X = arma::zeros(arma::size(tmp));
    for (int j = 0; j < X.n_cols; j++) {
        arma::vec v = tmp.col(j);
        int i = arma::index_max(v);
        X(i, j) = 1;
    }
    arma::mat mask = X;

    arma::mat At = arma::trans(A);
    arma::mat AtB = At * B;

    arma::mat AtA;
    if (compute_AtA) {
        AtA = At * A;
    }

    finish = now();
    t1 += duration(finish - start);

    arma::mat grad;
    arma::mat old_X = X;
    double min_grad_norm = 1e-3;
    for (int it = 0; it < max_iter; it++) {

        start = now();
        arma::mat AX = A * X;
        arma::mat obj = AX - B;

        if (compute_AtA) {
            grad = (AtA * X) - AtB;
        } else {
            grad = (At * (A * X)) - AtB;
        }
        arma::rowvec grad_norms = arma::sqrt(arma::sum(arma::square(grad)));
        arma::uvec unsaturated_cols = arma::find(grad_norms > min_grad_norm);
        if (unsaturated_cols.n_elem == 0) {
            break;
        }

        arma::urowvec s = arma::index_min(grad.cols(unsaturated_cols), 0);
        arma::urowvec v = arma::index_min(grad.cols(unsaturated_cols) % mask.cols(unsaturated_cols), 0);

        arma::mat S = arma::mat(
                arma::sp_mat(arma::join_vert(s, arma::regspace<arma::urowvec>(0, s.n_elem - 1)), arma::ones(s.n_elem),
                             X.n_rows, unsaturated_cols.n_elem));
        arma::mat V = arma::mat(
                arma::sp_mat(arma::join_vert(v, arma::regspace<arma::urowvec>(0, v.n_elem - 1)), arma::ones(v.n_elem),
                             X.n_rows, unsaturated_cols.n_elem));
        arma::mat D_FW = S - X.cols(unsaturated_cols);
        arma::mat D_A = X.cols(unsaturated_cols) - V;

        arma::rowvec delta =
                arma::sum(D_FW % grad.cols(unsaturated_cols)) - arma::sum(D_A % grad.cols(unsaturated_cols));

        finish = now();
        t2 += duration(finish - start);

        for (int k = 0; k < unsaturated_cols.n_elem; k++) {
            int j = unsaturated_cols(k);

            arma::vec d;
            double alpha_max = 1;
            if ((delta(k) <= 0) | (X(v(k), j) == 0)) {
                d = D_FW.col(k);
                alpha_max = 1;
                mask(s(k), j) = 1;
            } else {
                d = D_A.col(k);
                alpha_max = X(v(k), j) / (1 - X(v(k), j));
            }

            start = now();

            /*
                        // From: https://thatdatatho.com/gradient-descent-line-search-linear-regression/
                        vec g = grad.col(j);
                        double num = dot(g, g), denom, alpha = 0;
                        if( compute_AtA ) {
                            vec AtAg = AtA *  g;
                            denom = dot(g, AtAg);
                        } else {
                            vec Ag = A * g;
                            denom = dot(Ag, Ag);
                        }
                        alpha =  num / denom;
                        printf("\t%d- num = %.2e, denom = %.2e, Alpha = %.2e\n", it, num, denom, alpha);
            */

            double alpha = 2.0 / (it + 2.0);

            /*
                        vec q = A * d;
                        double alpha = 0;
                        double q_norm = norm(q, 1);
                        if(q_norm > 0) {
                            double q_norm_sq = q_norm*q_norm;
                            vec delta = -obj.col(j);
                            alpha = dot(q, delta) / q_norm_sq;
                        }
            */

            /*
            // Backtracking line-search
            vec g = grad.col(j);
            vec Ad = A * d;
            double e1 = dot(Ad, Ad);
            double alpha = 0;
            if(e1 != 0) {
                double e2 = 2 * dot(obj.col(j), Ad);
                double e3 = 0.5* dot(g, d); // multiplier can be in (0, 0.5]
                alpha = (e3 - e2) / e1;
            }

*/
            alpha = std::min(alpha, alpha_max);
            X.col(j) += alpha * d;
            if (0 < delta(k)) {
                if (X(v(k), j) < 1e-6)
                    mask(v(k), j) = 0;
            }
            finish = now();
            t5 += duration(finish - start);
        }

        start = now();
        //("%d- ", it);
        // double res = norm(old_X - X, "fro");
        // printf("%e\n", res);
        finish = now();
        t6 += duration(finish - start);
        /*
                if(res < min_diff) {
                    break;
                }
        */
        // old_X = X;
    }

    double total = t1 + t2 + t3 + t4 + t5 + t6;
    // printf("t1 = %3.f, t2 = %3.f, t3 = %3.f, t4 = %3.f, t5 = %3.f, t6 = %3.f\n", 100*t1/total, 100*t2/total, 100*t3/total, 100*t4/total, 100*t5/total, 100*t6/total);

    // X = clamp(X, 0, 1);
    // X = normalise(X, 1);

    return (X);
}
