#include "utils_internal/utils_decomp.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace actionet {

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

void orient_SVD(arma::field<arma::mat>& SVD_res) {
    arma::mat& U = SVD_res(0);
    arma::mat& V = SVD_res(2);
    int dim = static_cast<int>(arma::vec(SVD_res(1)).n_elem);

    for (int i = 0; i < dim; i++) {
        double n_up = 0, n_un = 0, n_vp = 0, n_vn = 0;
        for (arma::uword r = 0; r < U.n_rows; r++) {
            double val = U(r, i);
            if (val > 0) n_up += val * val;
            else         n_un += val * val;
        }
        for (arma::uword r = 0; r < V.n_rows; r++) {
            double val = V(r, i);
            if (val > 0) n_vp += val * val;
            else         n_vn += val * val;
        }
        if (std::sqrt(n_up) * std::sqrt(n_vp) < std::sqrt(n_un) * std::sqrt(n_vn)) {
            U.col(i) *= -1;
            V.col(i) *= -1;
        }
    }
}

void check_svd_axis_dimensions(arma::uword rows, arma::uword cols, const char* label) {
    constexpr arma::uword INT_MAX_UW = static_cast<arma::uword>(std::numeric_limits<int>::max());
    if (rows <= INT_MAX_UW && cols <= INT_MAX_UW) {
        return;
    }

    std::ostringstream msg;
    msg << label << ": matrix dimension exceeds INT_MAX "
        << "(rows=" << rows << ", cols=" << cols
        << ", INT_MAX=" << INT_MAX_UW << "). "
        << "Per-axis dimensions above INT_MAX (~2.1B) are not yet supported "
           "by any SVD algorithm.";
    throw std::overflow_error(msg.str());
}

void clamp_halko_dim(arma::uword rows, arma::uword cols, int& dim) {
    const arma::uword mn_min = std::min(rows, cols);
    if (dim < 1) {
        dim = 1;
    }
    if (static_cast<arma::uword>(dim) + 2 > mn_min) {
        dim = static_cast<int>(mn_min) - 2;
    }
    if (dim < 1) {
        dim = 1;
    }
}

} // namespace actionet
