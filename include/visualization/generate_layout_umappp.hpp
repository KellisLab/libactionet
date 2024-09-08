#ifndef ACTIONET_GENERATE_LAYOUT_HPP
#define ACTIONET_GENERATE_LAYOUT_HPP

#include "libactionet_config.hpp"

// Constants
//#define NEGATIVE_SAMPLE_RATE 3.0
//#define ADAM_ALPHA 1.0 /*same as learning_rate*/
//#define ADAM_BETA1 0.5 /*only adam: between 0 and 1*/
//#define ADAM_BETA2 0.9 /*only adam: between 0 and 1*/
//#define ADAM_EPS 1e-7  /*only adam: between 1e-8 and 1e-3*/

// Exported functions
namespace actionet {
    arma::mat layoutNetwork_umappp(arma::sp_mat& G, arma::mat& initial_embedding, int thread_no = 0);
} // namespace actionet

#endif //ACTIONET_GENERATE_LAYOUT_HPP
