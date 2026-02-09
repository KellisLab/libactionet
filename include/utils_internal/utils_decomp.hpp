#ifndef ACTIONET_UTILS_DECOMP_HPP
#define ACTIONET_UTILS_DECOMP_HPP

#include "libactionet_config.hpp"
#include "aarand/aarand.hpp"

/// @brief Orthogonalize X against Y using temporary storage.
void orthog(double *X, double *Y, double *T, int xm, int xn, int yn);

/// @brief Convergence tests for iterative SVD.
void convtests(int Bsz, int n, double tol, double svtol, double Smax, double *svratio, double *residuals, int *k,
               int *converged, double S);

/// @brief Gram-Schmidt orthogonalization.
void gram_schmidt(arma::mat &A);

/// @brief Generate standard normal random matrix.
arma::mat randNorm(int l, int m, int seed);

/// @brief Compute SVD via eigendecomposition.
arma::field<arma::mat> eigSVD(const arma::mat &A);

/// @brief Orient SVD components for consistency.
arma::field<arma::mat> orient_SVD(arma::field<arma::mat> SVD_res);

// Functions: inline
/// @brief Fill a buffer with standard normal values.
inline void StdNorm(double *v, int n, std::mt19937_64 engine) {
    for (int ii = 0; ii < n - 1; ii += 2) {
        auto paired = aarand::standard_normal(engine);
        v[ii] = paired.first;
        v[ii + 1] = paired.second;
    }
    auto paired = aarand::standard_normal(engine);
    // Final value if length(v) is odd. Overwrites last value otherwise.
    v[n - 1] = paired.first;
}

#endif //ACTIONET_UTILS_DECOMP_HPP
