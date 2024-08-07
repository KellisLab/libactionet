#ifndef ACTIONET_HPP
#define ACTIONET_HPP

#include "config_arma.hpp"
#include "config_actionet.hpp"

namespace ACTIONet
{
  // build_network
  // COnstruct k-nn network
  arma::sp_mat buildNetwork(arma::mat H, std::string algorithm, std::string distance_metric, double density, int thread_no, double M,
                            double ef_construction, double ef, bool mutual_edges_only, int k);

} // namespace ACTIONet

#endif
