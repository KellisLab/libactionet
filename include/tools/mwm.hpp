// Maximum weight matching algorithm implementations
#ifndef ACTIONET_MWM_HPP
#define ACTIONET_MWM_HPP

#include "libactionet_config.hpp"

// Functions: private
/// @brief Internal driver for maximum weight matching.
double MWM_driver(int n, int m, int nedges, double* vv1, double* vv2, double* weight, double* out1, double* out2,
                  int* noutedges);

// Exported
namespace actionet {
    /// @brief Maximum weight matching using the Hungarian algorithm.
    ///
    /// @param G Dense weight matrix.
    /// @return Matching matrix (indices/weights encoded by implementation).
    arma::mat MWM_hungarian(arma::mat& G);

    /// @brief Low-rank spectral network alignment (rank-1).
    ///
    /// @param u Left vector.
    /// @param v Right vector.
    /// @param u_threshold Threshold for u support.
    /// @param v_threshold Threshold for v support.
    /// @return 2 x k matrix of matched indices.
    arma::umat MWM_rank1(const arma::vec& u, const arma::vec& v, double u_threshold, double v_threshold);
} // namespace actionet

#endif //ACTIONET_MWM_HPP
