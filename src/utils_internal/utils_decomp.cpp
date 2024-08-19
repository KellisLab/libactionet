#include "utils_internal/utils_decomp.hpp"

void orthog(double *X, double *Y, double *T, int xm, int xn, int yn) {
    double a = 1, b = 1;
    int inc = 1;
    std::memset(T, 0, xn * yn * sizeof(double));
    // T = t(X) * Y
    cblas_dgemv(CblasColMajor, CblasTrans, xm, xn, a, X, xm, Y, inc, b, T, inc);
    // Y = Y - X * T
    a = -1.0;
    b = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, xm, xn, a, X, xm, T, inc, b, Y, inc);
}

// Convergence test
void convtests(int Bsz, int n, double tol, double svtol, double Smax, double *svratio, double *residuals, int *k,
               int *converged, double S) {
    int Len_res = 0;
    for (int j = 0; j < Bsz; j++) {
        if ((std::fabs(residuals[j]) < tol * Smax) && (svratio[j] < svtol))
            Len_res++;
    }

    if (Len_res >= n || S == 0) {
        *converged = 1;
        return;
    }
    if (*k < n + Len_res)
        *k = n + Len_res;

    if (*k > Bsz - 3)
        *k = Bsz - 3;

    if (*k < 1)
        *k = 1;

    *converged = 0;
}

void gram_schmidt(arma::mat &A) {
    for (arma::uword i = 0; i < A.n_cols; ++i) {
        for (arma::uword j = 0; j < i; ++j) {
            double r = arma::dot(A.col(i), A.col(j));
            A.col(i) -= r * A.col(j);
        }

        double col_norm = arma::norm(A.col(i), 2);

        if (col_norm < 1E-4) {
            for (arma::uword k = i; k < A.n_cols; ++k)
                A.col(k).zeros();

            return;
        }
        A.col(i) /= col_norm;
    }
}

arma::mat randNorm(int l, int m, int seed) {
    std::default_random_engine gen(seed);
    std::normal_distribution<double> normDist(0.0, 1.0);

    arma::mat R(l, m);
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < l; i++) {
            R(i, j) = normDist(gen);
        }
    }
    return R;
}

arma::field<arma::mat> eigSVD(const arma::mat &A) {
    int n = A.n_cols;
    arma::mat B = arma::trans(A) * A;

    arma::vec d;
    arma::mat V;
    arma::eig_sym(d, V, B);
    d = sqrt(d);

    // Compute U
    arma::sp_mat S(n, n);
    S.diag() = 1 / d;
    arma::mat U = (S * arma::trans(V)) * arma::trans(A);
    U = arma::trans(U);

    arma::field<arma::mat> out(3);

    out(0) = U;
    out(1) = d;
    out(2) = V;

    return (out);
}

arma::field<arma::mat> orient_SVD(arma::field<arma::mat> SVD_res) {
    arma::mat U = SVD_res(0);
    arma::vec s = SVD_res(1);
    arma::mat V = SVD_res(2);

    int dim = s.n_elem;
    arma::uvec mask_idx;

    for (int i = 0; i < dim; i++) {
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
            U.col(i) *= -1;
            V.col(i) *= -1;
        }
    }

    arma::field<arma::mat> out(3);
    out(0) = U;
    out(1) = s;
    out(2) = V;

    return (out);
}
