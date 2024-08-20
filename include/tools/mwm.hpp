// Maximum weight matching algorithm implementations
#ifndef ACTIONET_MWM_HPP
#define ACTIONET_MWM_HPP

#include "libactionet_config.hpp"

// Functions: private
double MWM_driver(int n, int m, int nedges, double *vv1, double *vv2, double *weight, double *out1, double *out2,
                  int *noutedges);

// Exported
namespace actionet {

    arma::mat MWM_hungarian(arma::mat &G);

    // Low Rank Spectral Network Alignment
    // (https://dl.acm.org/citation.cfm?doid=3178876.3186128)
    arma::umat MWM_rank1(const arma::vec& u, const arma::vec& v, double u_threshold, double v_threshold);

} // namespace actionet

#endif //ACTIONET_MWM_HPP
