#ifndef ACTIONET_AUTOCORRELATION_HPP
#define ACTIONET_AUTOCORRELATION_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // G is the symmetric adjacency matrix, scores is a nodes x features matrix
    arma::field<arma::vec>
        autocorrelation_Moran_parametric(arma::mat G, const arma::mat& scores, int normalization_method = 4,
                                         int thread_no = 0);

    arma::field<arma::vec>
        autocorrelation_Moran_parametric(arma::sp_mat G, const arma::mat& scores, int normalization_method = 4,
                                         int thread_no = 0);

    arma::field<arma::vec>
        autocorrelation_Moran(arma::mat G, const arma::mat& scores, int normalization_method = 1, int perm_no = 30,
                              int thread_no = 0);

    arma::field<arma::vec>
        autocorrelation_Moran(arma::sp_mat G, const arma::mat& scores, int normalization_method = 1, int perm_no = 30,
                              int thread_no = 0);

    arma::field<arma::vec>
        autocorrelation_Geary(const arma::mat& G, const arma::mat& scores, int normalization_method = 1,
                              int perm_no = 30,
                              int thread_no = 0);

    arma::field<arma::vec>
        autocorrelation_Geary(const arma::sp_mat& G, const arma::mat& scores, int normalization_method = 1,
                              int perm_no = 30,
                              int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_AUTOCORRELATION_HPP
