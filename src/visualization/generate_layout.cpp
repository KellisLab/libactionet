#include "visualization/generate_layout.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "visualization/create_xmap.hpp"
#include "utils_internal/utils_stats.hpp"
#include "visualization/find_ab.hpp"
#include "tools/normalization.hpp"
#include "uwot/coords.h"
#include "colorspace.h"
#include <cfloat>

arma::sp_mat smoothKNN(arma::sp_mat &D, int max_iter, double epsilon, double bandwidth, double local_connectivity,
                       double min_k_dist_scale, double min_sim, int thread_no) {

    int nV = D.n_cols;
    arma::sp_mat G = D;

    //#pragma omp parallel for num_threads(thread_no)
    for (int i = 0; i < nV; i++) {
        //  ParallelFor(0, nV, thread_no, [&](size_t i, size_t threadId) {
        arma::sp_mat v = D.col(i);
        arma::vec vals = nonzeros(v);
        if (vals.n_elem > local_connectivity) {
            double rho = min(vals);
            arma::vec negated_shifted_vals = -(vals - rho);
            arma::uvec deflate = arma::find(vals <= rho);
            negated_shifted_vals(deflate).zeros();

            double target = std::log2(vals.n_elem + 1);

            // Binary search to find optimal sigma
            double sigma = 1.0;
            double lo = 0.0;
            double hi = DBL_MAX;

            int j;
            for (j = 0; j < max_iter; j++) {
                double obj = sum(exp(negated_shifted_vals / sigma));

                if (abs(obj - target) < epsilon) {
                    break;
                }

                if (target < obj) {
                    hi = sigma;
                    sigma = 0.5 * (lo + hi);
                } else {
                    lo = sigma;
                    if (hi == DBL_MAX) {
                        sigma *= 2;
                    } else {
                        sigma = 0.5 * (lo + hi);
                    }
                }
            }

            // double obj = sum(exp(negated_shifted_vals / sigma));
            // TODO: This is obviously a bug. `mean_dist` does not exist.
            double mean_dist = arma::mean(mean_dist);
            sigma = std::max(min_k_dist_scale * mean_dist, sigma);

            for (arma::sp_mat::col_iterator it = G.begin_col(i); it != G.end_col(i); ++it) {
                *it = std::max(min_sim, std::exp(-std::max(0.0, (*it) - rho) / (sigma * bandwidth)));
            }
        } else {
            for (arma::sp_mat::col_iterator it = G.begin_col(i); it != G.end_col(i); ++it) {
                *it = 1.0;
            }
        }
    }

    return (G);
}

namespace ACTIONet {

    arma::field<arma::mat> layoutNetwork_xmap(arma::sp_mat &G, arma::mat &initial_position, bool presmooth_network,
                                              const std::string &method, double min_dist, double spread, double gamma,
                                              unsigned int n_epochs, int thread_no, int seed, double learning_rate,
                                              int sim2dist) {
        if (thread_no <= 0) {
            thread_no = SYS_THREADS_DEF;
        }

        auto found = find_ab(spread, min_dist);
        double a = found.first;
        double b = found.second;

        stdout_printf(
                "Laying-out input network: method = %s, a = %.3f, b = %.3f (epochs = %d, threads=%d)\n",
                method.c_str(), a, b, n_epochs, thread_no);

        bool move_other = true;
        std::size_t grain_size = 1;
        bool pcg_rand = true;
        bool approx_pow = true;
        bool batch = true;
        std::string opt_name = "adam";
        double alpha = ADAM_ALPHA, beta1 = ADAM_BETA1, beta2 = ADAM_BETA2,
                eps = ADAM_EPS, negative_sample_rate = NEGATIVE_SAMPLE_RATE;

        arma::field<arma::mat> res(3);
        std::mt19937_64 engine(seed);

        arma::mat init_coors;
        if (initial_position.n_rows != G.n_rows) {
            stderr_printf("Number of rows in the initial_position should match with the number of vertices in G\n");
            FLUSH;
            return (res);
        }

        // Encode positive edges of the graph
        arma::sp_mat H = G;
        if (presmooth_network == true) {
            stdout_printf("%\tSmoothing similarities (sim2dist = %d) ... ", sim2dist);
            if (sim2dist == 1) {
                H.for_each([](arma::sp_mat::elem_type &val) { val = 1 - val; });
            } else if (sim2dist == 2) {
                H.for_each([](arma::sp_mat::elem_type &val) { val = (1 - val) * (1 - val); });
            } else if (sim2dist == 3) {
                H.for_each([](arma::sp_mat::elem_type &val) { val = -std::log(val); });
            } else {
                H.for_each([](arma::sp_mat::elem_type &val) { val = 1 / val; });
            }

            int max_iter = 64;
            double epsilon = 1e-6, bandwidth = 1.0, local_connectivity = 1.0,
                    min_k_dist_scale = 1e-3, min_sim = 1e-8;

            H = smoothKNN(H, max_iter, epsilon, bandwidth, local_connectivity,
                          min_k_dist_scale, min_sim, thread_no);
            stdout_printf("done\n");
        }

        double w_max = arma::max(arma::max(H));
        H.clean(w_max / n_epochs);

        arma::sp_mat Ht = arma::trans(H);
        Ht.sync();

        unsigned int nV = H.n_rows;
        unsigned int nE = H.n_nonzero;
        unsigned int nD = init_coors.n_rows;

        std::vector<unsigned int> positive_head(nE);
        std::vector<unsigned int> positive_tail(nE);
        std::vector<float> epochs_per_sample(nE);

        std::vector<unsigned int> positive_ptr(Ht.n_cols + 1);

        int i = 0;
        if (batch == false) {
            for (arma::sp_mat::iterator it = H.begin(); it != H.end(); ++it) {
                epochs_per_sample[i] = w_max / (*it);
                positive_head[i] = it.row();
                positive_tail[i] = it.col();
                i++;
            }
        } else {
            for (arma::sp_mat::iterator it = Ht.begin(); it != Ht.end(); ++it) {
                epochs_per_sample[i] = w_max / (*it);
                positive_tail[i] = it.row();
                positive_head[i] = it.col();
                i++;
            }
            for (int k = 0; k < Ht.n_cols + 1; k++) {
                positive_ptr[k] = Ht.col_ptrs[k];
            }
        }

        arma::mat coors2D = initial_position.cols(0, 1);
        init_coors = arma::trans(zscore(coors2D));

        // Initial coordinates of vertices (0-simplices)
        std::vector<float> head_embedding(init_coors.n_elem);
        arma::fmat sub_coor = arma::conv_to<arma::fmat>::from(init_coors);
        std::memcpy(head_embedding.data(), sub_coor.memptr(),
                    sizeof(float) * head_embedding.size());
        // vector<float> tail_embedding(head_embedding);
        uwot::Coords coords = uwot::Coords(head_embedding);

        UmapFactory umap_factory(
                move_other, pcg_rand, coords.get_head_embedding(),
                coords.get_tail_embedding(), positive_head, positive_tail, positive_ptr,
                n_epochs, nV, nV, epochs_per_sample, learning_rate, negative_sample_rate,
                batch, thread_no, grain_size, opt_name, alpha, beta1, beta2, eps, engine);

        stdout_printf("Computing 2D layout ... ");
        FLUSH;
        if (method == "umap") {
            create_umap(umap_factory, a, b, gamma, approx_pow);
        } else if (method == "tumap") {
            create_tumap(umap_factory);
        } else if (method == "largevis") {
            create_largevis(umap_factory, gamma);
        } else if (method == "pacmap") {
            create_pacmap(umap_factory, a, b);
        } else {
            stderr_printf("Unknown method: %s\n", method.c_str());
            FLUSH;
            return (res);
        }
        arma::fmat coordinates_float(coords.get_head_embedding().data(), 2, nV);
        arma::mat coordinates_2D = arma::trans(arma::conv_to<arma::mat>::from(coordinates_float));
        stdout_printf("done\n");
        FLUSH;

        arma::mat coordinates_3D = arma::zeros(nV, 3);
        arma::mat RGB_colors = arma::zeros(nV, 3);
        if (initial_position.n_cols > 2) {
            arma::mat coors3D = arma::join_rows(coordinates_2D, initial_position.col(2));
            init_coors = arma::trans(zscore(coors3D));
            head_embedding.clear();
            head_embedding.resize(nV * 3);
            sub_coor = arma::conv_to<arma::fmat>::from(init_coors);
            std::memcpy(head_embedding.data(), sub_coor.memptr(),
                        sizeof(float) * head_embedding.size());
            coords = uwot::Coords(head_embedding);

            UmapFactory umap_factory_3D(
                    move_other, pcg_rand, coords.get_head_embedding(),
                    coords.get_tail_embedding(), positive_head, positive_tail, positive_ptr,
                    n_epochs / 2, nV, nV, epochs_per_sample, learning_rate,
                    negative_sample_rate, batch, thread_no, grain_size, opt_name, alpha,
                    beta1, beta2, eps, engine);

            stdout_printf("Computing 3D layout ... ");
            FLUSH;
            if (method == "umap") {
                create_umap(umap_factory_3D, a, b, gamma, approx_pow);
            } else if (method == "tumap") {
                create_tumap(umap_factory_3D);
            } else if (method == "largevis") {
                create_largevis(umap_factory_3D, gamma);
            } else if (method == "pacmap") {
                create_pacmap(umap_factory_3D, a, b);
            } else {
                stderr_printf("Unknown method: %s\n", method.c_str());
                FLUSH;
                return (res);
            }

            arma::fmat coordinates_float(coords.get_head_embedding().data(), 3, nV);
            coordinates_3D = arma::trans(arma::conv_to<arma::mat>::from(coordinates_float));

            stdout_printf("done\n");
            FLUSH;

            stdout_printf("Computing de novo node colors ... ");
            FLUSH;

            arma::mat U;
            arma::vec s;
            arma::mat V;
            arma::svd_econ(U, s, V, coordinates_3D, "left", "std");

            arma::mat Z = normalize_scores(U.cols(0, 2), 1, thread_no);

            arma::vec a = 75 * Z.col(0);
            arma::vec b = 75 * Z.col(1);

            arma::vec L = Z.col(2);
            L = 25.0 + 70.0 * (L - arma::min(L)) / (arma::max(L) - arma::min(L));

            double r_channel, g_channel, b_channel;
            for (int i = 0; i < nV; i++) {
                Lab2Rgb(&r_channel, &g_channel, &b_channel, L(i), a(i), b(i));

                RGB_colors(i, 0) = std::min(1.0, std::max(0.0, r_channel));
                RGB_colors(i, 1) = std::min(1.0, std::max(0.0, g_channel));
                RGB_colors(i, 2) = std::min(1.0, std::max(0.0, b_channel));
            }
            stdout_printf("done\n");
            FLUSH;
        }

        res(0) = coordinates_2D;
        res(1) = coordinates_3D;
        res(2) = RGB_colors;

        return (res);
    }

} // namespace ACTIONet