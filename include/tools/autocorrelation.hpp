#ifndef LIBACTIONET_AUTOCORRELATION_HPP
#define LIBACTIONET_AUTOCORRELATION_HPP

#include "libactionet_config.hpp"
#include "tools/normalization.hpp"

// Exported
namespace ACTIONet {

    // G is the symmetric adjacency matrix, scores is a nodes x features matrix
    arma::field<arma::vec>
    autocorrelation_Moran_parametric(arma::mat G, arma::mat scores, int normalization_method = 4, int thread_no = 0);

    arma::field<arma::vec>
    autocorrelation_Moran_parametric(arma::sp_mat G, arma::mat scores, int normalization_method = 4, int thread_no = 0);

    arma::field<arma::vec>
    autocorrelation_Moran(arma::mat G, arma::mat scores, int normalization_method = 1, int perm_no = 30,
                          int thread_no = 0);

    arma::field<arma::vec>
    autocorrelation_Moran(arma::sp_mat G, arma::mat scores, int normalization_method = 1, int perm_no = 30,
                          int thread_no = 0);

    arma::field<arma::vec>
    autocorrelation_Geary(arma::mat G, arma::mat scores, int normalization_method = 1, int perm_no = 30,
                          int thread_no = 0);

    arma::field<arma::vec>
    autocorrelation_Geary(arma::sp_mat G, arma::mat scores, int normalization_method = 1, int perm_no = 30,
                          int thread_no = 0);

} // namespace ACTIONet

#endif //LIBACTIONET_AUTOCORRELATION_HPP
