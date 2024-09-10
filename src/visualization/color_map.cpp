#include "visualization/color_map.hpp"
#include "tools/normalization.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "colorspace.h"

namespace actionet {
    arma::mat computeNodeColors(const arma::mat& coordinates, int thread_no) {
        // try {
            if (coordinates.n_cols < 3) {
                throw std::invalid_argument("Invalid matrix size. `coordinates.n_cols` must be >= 3");
            }

        // }
        // catch (const std::invalid_argument& e) {
        //     stdout_printf(e.what());
        // }

        stdout_printf("Computing node colors ... ");
        FLUSH;

        int threads_use = get_num_threads(SYS_THREADS_DEF, thread_no);
        arma::mat U;
        arma::vec s;
        arma::mat V;
        arma::svd_econ(U, s, V, coordinates, "left", "std");

        arma::mat Z = normalize_scores(U.cols(0, 2), 1, threads_use);

        arma::vec a = 75 * Z.col(0);
        arma::vec b = 75 * Z.col(1);
        arma::vec L = Z.col(2);
        L = 25.0 + 70.0 * (L - arma::min(L)) / (arma::max(L) - arma::min(L));

        arma::mat rgb_cols = arma::zeros(coordinates.n_rows, 3);
        double r_channel, g_channel, b_channel;

        // #pragma omp parallel for num_threads(threads_use)
        for (int i = 0; i < rgb_cols.n_rows; i++) {
            Lab2Rgb(&r_channel, &g_channel, &b_channel, L(i), a(i), b(i));

            rgb_cols(i, 0) = std::min(1.0, std::max(0.0, r_channel));
            rgb_cols(i, 1) = std::min(1.0, std::max(0.0, g_channel));
            rgb_cols(i, 2) = std::min(1.0, std::max(0.0, b_channel));
        }
        stdout_printf("done\n");
        FLUSH;

        return (rgb_cols);
    }
} // actionet
