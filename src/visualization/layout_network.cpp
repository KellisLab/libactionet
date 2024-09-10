#include "visualization/layout_network.hpp"
#include "visualization/uwot_actionet.hpp"
#include "utils_internal/utils_parallel.hpp"

namespace actionet {
    arma::mat layoutNetwork(arma::sp_mat& G, arma::mat& initial_coordinates, std::string method, unsigned int n_components,
                            float spread, float min_dist, unsigned int n_epochs, float learning_rate,
                            float repulsion_strength, float negative_sample_rate, bool approx_pow, bool pcg_rand,
                            bool batch, unsigned int grain_size, int seed, int thread_no, bool verbose, float a,
                            float b, std::string opt_method, float alpha, float beta1, float beta2, float eps) {
        unsigned int n_threads = get_num_threads(SYS_THREADS_DEF, thread_no);

        alpha = (alpha == -1) ? learning_rate : alpha;
        // beta1 = (beta1 == -1) ? ADAM_BETA1 : beta1;
        // beta2 = (beta2 == -1) ? ADAM_BETA2 : beta2;
        // eps = (eps == -1) ? ADAM_EPS : eps;

        OptimizerArgs opt_args = OptimizerArgs(opt_method, alpha, beta1, beta2, eps);

        UwotArgs uwot_args(method, n_components, spread, min_dist, n_epochs, learning_rate, repulsion_strength,
                           negative_sample_rate, approx_pow, pcg_rand, batch, seed, n_threads, grain_size, verbose,
                           opt_args);

        if (a != 0 || b != 0) {
            uwot_args.set_ab(a, b);
        }

        arma::mat coords_out = optimize_layout_uwot(G, initial_coordinates, uwot_args);
        return (coords_out);
    }
} // namespace actionet
