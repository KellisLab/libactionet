#include "visualization/generate_layout.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "visualization/create_xmap.hpp"
#include "utils_internal/utils_stats.hpp"
#include "visualization/find_ab.hpp"
#include "tools/normalization.hpp"
#include "uwot/coords.h"
#include "colorspace.h"
#include <cfloat>

namespace actionet {
    arma::field<arma::mat> layoutNetwork_xmap(arma::sp_mat& G, arma::mat& initial_position, const std::string& method,
                                              double min_dist, double spread, double gamma, unsigned int n_epochs,
                                              double learning_rate, int seed, int thread_no) {
        int threads_use = get_num_threads(SYS_THREADS_DEF, thread_no);
        auto found = find_ab(spread, min_dist);
        double a = found.first;
        double b = found.second;

        stdout_printf(
            "Laying-out input network: method = %s, a = %.3f, b = %.3f (epochs = %d, threads=%d)\n",
            method.c_str(), a, b, n_epochs, threads_use);

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

        double w_max = arma::max(arma::max(H));
        H.clean(w_max / n_epochs);

        arma::sp_mat Ht = arma::trans(H);
        Ht.sync();

        unsigned int nV = H.n_rows;
        unsigned int nE = H.n_nonzero;

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
        }
        else {
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
        std::memcpy(head_embedding.data(), sub_coor.memptr(), sizeof(float) * head_embedding.size());
        uwot::Coords coords = uwot::Coords(head_embedding);

        UmapFactory umap_factory(
            move_other, pcg_rand, coords.get_head_embedding(),
            coords.get_tail_embedding(), positive_head, positive_tail, positive_ptr,
            n_epochs, nV, nV, epochs_per_sample, learning_rate, negative_sample_rate,
            batch, threads_use, grain_size, opt_name, alpha, beta1, beta2, eps, engine);

        stdout_printf("Computing 2D layout ... ");
        FLUSH;
        if (method == "umap") {
            create_umap(umap_factory, a, b, gamma, approx_pow);
        }
        else if (method == "tumap") {
            create_tumap(umap_factory);
        }
        else if (method == "largevis") {
            create_largevis(umap_factory, gamma);
        }
        else if (method == "pacmap") {
            create_pacmap(umap_factory, a, b);
        }
        else {
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
                negative_sample_rate, batch, threads_use, grain_size, opt_name, alpha,
                beta1, beta2, eps, engine);

            stdout_printf("Computing 3D layout ... ");
            FLUSH;
            if (method == "umap") {
                create_umap(umap_factory_3D, a, b, gamma, approx_pow);
            }
            else if (method == "tumap") {
                create_tumap(umap_factory_3D);
            }
            else if (method == "largevis") {
                create_largevis(umap_factory_3D, gamma);
            }
            else if (method == "pacmap") {
                create_pacmap(umap_factory_3D, a, b);
            }
            else {
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

            arma::mat Z = normalize_scores(U.cols(0, 2), 1, threads_use);

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
} // namespace actionet
