#include "visualization/optimize_layout.hpp"
#include "visualization/UmapFactory.hpp"
#include "utils_internal/utils_parallel.hpp"
// #include "visualization/create_xmap.hpp"
// #include "visualization/find_ab.hpp"
// #include "tools/normalization.hpp"
// #include "colorspace.h"
// #include <cfloat>
// #include <visualization/generate_layout.hpp>

void verboseStatus(const UwotArgs& method_args) {
    stderr_printf("Optimizing layout using method '%s': %d components \n", method_args.method.c_str(),
                  method_args.n_components);
    switch (method_args.get_cost_func()) {
        case METHOD_UMAP:
        case METHOD_TUMAP:
            stderr_printf("UMAP embedding parameters a = %.3f, b = %.3f' \n", method_args.a, method_args.b);
            break;
    }
    stderr_printf("Optimizing for %d epochs with %d threads \n", method_args.n_epochs, (int)method_args.n_threads);
    FLUSH;
};

void create_umap(UmapFactory& umap_factory, const UwotArgs& method_args) {
    if (method_args.approx_pow) {
        const uwot::apumap_gradient gradient(method_args.a, method_args.b, method_args.gamma);
        umap_factory.create(gradient);
    }
    else {
        const uwot::umap_gradient gradient(method_args.a, method_args.b, method_args.gamma);
        umap_factory.create(gradient);
    }
}

void create_tumap(UmapFactory& umap_factory) {
    const uwot::tumap_gradient gradient;
    umap_factory.create(gradient);
}

void create_largevis(UmapFactory& umap_factory, const UwotArgs& method_args) {
    const uwot::largevis_gradient gradient(method_args.gamma);
    umap_factory.create(gradient);
}


uwot::Coords getCoords(arma::mat& initial_position, int n_components) {
    arma::mat init_coors = arma::trans(initial_position.cols(0, n_components - 1));

    // Initial coordinates of vertices (0-simplices)
    std::vector<float> head_embedding(init_coors.n_elem);
    arma::fmat sub_coor = arma::conv_to<arma::fmat>::from(init_coors);
    std::memcpy(head_embedding.data(), sub_coor.memptr(), sizeof(float) * head_embedding.size());
    uwot::Coords coords = uwot::Coords(head_embedding);

    return coords;
}

UmapFactory buildFactory(arma::sp_mat& G, arma::mat& initial_position, const UwotArgs& uwot_args) {
    uwot::Coords coords = getCoords(initial_position, uwot_args.n_components);

    bool move_other = true;
    arma::sp_mat H = G;
    double w_max = arma::max(arma::max(H));
    H.clean(w_max / uwot_args.n_epochs);

    arma::sp_mat Ht = arma::trans(H); // TODO: .eval()
    Ht.sync();

    unsigned int nV = H.n_rows;
    unsigned int nE = H.n_nonzero;

    std::vector<unsigned int> positive_head(nE);
    std::vector<unsigned int> positive_tail(nE);
    std::vector<float> epochs_per_sample(nE);
    std::vector<unsigned int> positive_ptr(Ht.n_cols + 1);

    int i = 0;
    if (uwot_args.batch == false) {
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

    UmapFactory UF(move_other,
                   uwot_args.pcg_rand,
                   coords.get_head_embedding(), coords.get_tail_embedding(),
                   positive_head, positive_tail, positive_ptr,
                   uwot_args.n_epochs,
                   nV, nV,
                   epochs_per_sample, uwot_args.alpha,
                   uwot_args.opt_args, uwot_args.negative_sample_rate, uwot_args.batch,
                   uwot_args.n_threads, uwot_args.grain_size, uwot_args.verbose);

    return (UF);
}

namespace actionet {
    arma::mat optimize_layout_uwot(arma::sp_mat& G, arma::mat& initial_position, UwotArgs uwot_args) {
        uwot_args.n_threads = get_num_threads(SYS_THREADS_DEF, static_cast<int>(uwot_args.n_threads));

        if (uwot_args.n_epochs == 0) {
            uwot_args.n_epochs = (initial_position.n_rows <= 10000) ? 500 : 200;
        }

        // if (initial_position.n_rows != G.n_rows) {
        //     stderr_printf("Number of rows in the initial_position should match with the number of vertices in G\n");
        //     FLUSH;
        //     return (arma::field<arma::mat>(3));
        // }

        //
        // auto found = find_ab(spread, min_dist);
        // double a = found.first;
        // double b = found.second;

        // bool move_other = true;
        // std::size_t grain_size = 1;
        // bool pcg_rand = true;
        // bool approx_pow = true;
        // bool batch = true;
        // std::string opt_name = "adam";
        // double alpha = OPT_ALPHA, beta1 = ADAM_BETA1, beta2 = ADAM_BETA2,
        //        eps = ADAM_EPS, negative_sample_rate = NEGATIVE_SAMPLE_RATE;

        // std::mt19937_64 engine(seed);

        // stdout_printf(
        //     "Laying-out input network: method = %s, a = %.3f, b = %.3f (epochs = %d, threads=%d)\n",
        //     uwot_args.method.c_str(), uwot_args.a, uwot_args.b, uwot_args.n_epochs, threads_use);


        // Encode positive edges of the graph
        // arma::sp_mat H = G;
        // double w_max = arma::max(arma::max(H));
        // H.clean(w_max / n_epochs);
        //
        // arma::sp_mat Ht = arma::trans(H);
        // Ht.sync();
        //
        // unsigned int nV = H.n_rows;
        // unsigned int nE = H.n_nonzero;
        //
        // std::vector<unsigned int> positive_head(nE);
        // std::vector<unsigned int> positive_tail(nE);
        // std::vector<float> epochs_per_sample(nE);
        //
        // std::vector<unsigned int> positive_ptr(Ht.n_cols + 1);
        //
        // int i = 0;
        // if (batch == false) {
        //     for (arma::sp_mat::iterator it = H.begin(); it != H.end(); ++it) {
        //         epochs_per_sample[i] = w_max / (*it);
        //         positive_head[i] = it.row();
        //         positive_tail[i] = it.col();
        //         i++;
        //     }
        // }
        // else {
        //     for (arma::sp_mat::iterator it = Ht.begin(); it != Ht.end(); ++it) {
        //         epochs_per_sample[i] = w_max / (*it);
        //         positive_tail[i] = it.row();
        //         positive_head[i] = it.col();
        //         i++;
        //     }
        //     for (int k = 0; k < Ht.n_cols + 1; k++) {
        //         positive_ptr[k] = Ht.col_ptrs[k];
        //     }
        // }

        // uwot::Coords coords = getCoords(initial_position, 2);
        //
        // UmapFactory umap_factory(
        //     move_other, pcg_rand, coords.get_head_embedding(),
        //     coords.get_tail_embedding(), positive_head, positive_tail, positive_ptr,
        //     n_epochs, nV, nV, epochs_per_sample, learning_rate, negative_sample_rate,
        //     batch, threads_use, grain_size, opt_name, alpha, beta1, beta2, eps, engine);

        UmapFactory umap_factory = buildFactory(G, initial_position, uwot_args);

        // stdout_printf("Computing 2D layout ... ");


        if (uwot_args.verbose) { verboseStatus(uwot_args); }

        switch (uwot_args.get_cost_func()) {
            case METHOD_TUMAP:
                create_tumap(umap_factory);
                break;
            case METHOD_LARGEVIZ:
                create_largevis(umap_factory, uwot_args);
                break;
            case METHOD_UMAP:
            default:
                create_umap(umap_factory, uwot_args);
        }

        // if (method == "umap") {
        //     create_umap(umap_factory, a, b, gamma, approx_pow);
        // }
        // else if (method == "tumap") {
        //     create_tumap(umap_factory);
        // }
        // else if (method == "largevis") {
        //     create_largevis(umap_factory, gamma);
        // }
        // else {
        //     stderr_printf("Unknown method: %s\n", method.c_str());
        //     FLUSH;
        //     return (res);
        // }

        // arma::fmat coordinates_float(coords.get_head_embedding().data(), 2, nV);

        arma::fmat uwot_embedding(umap_factory.head_embedding.data(), uwot_args.n_components, G.n_rows);
        arma::mat coords_out = arma::trans(arma::conv_to<arma::mat>::from(uwot_embedding));
        stderr_printf("Optimization finished\n");
        FLUSH;

        return (coords_out);
    }
} // namespace actionet
