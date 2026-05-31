#include "visualization/uwot_actionet.hpp"
#include "visualization/UmapFactory.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "uwot/coords.h"

#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif

namespace {

struct EdgeVectors {
    std::vector<unsigned int> positive_head;
    std::vector<unsigned int> positive_tail;
    std::vector<float> epochs_per_sample;
    std::vector<unsigned int> positive_ptr;
    unsigned int n_vertices;
};

const char* env_or_unset(const char* key) {
    const char* value = std::getenv(key);
    return (value && value[0] != '\0') ? value : "<unset>";
}

unsigned int affinity_cpu_count() {
#if defined(__linux__) && !defined(__ANDROID__)
    cpu_set_t cpuset;
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        return static_cast<unsigned int>(CPU_COUNT(&cpuset));
    }
#endif
    return 0;
}

const char* optimizer_name(const OptimizerArgs& opt_args) {
    return opt_args.opt_method == OPT_METHOD_SGD ? "sgd" : "adam";
}

void verboseStatus(const UwotArgs& method_args, std::size_t requested_threads) {
    stderr_printf("Optimizing layout using method '%s': %d components \n", method_args.get_method().c_str(),
                  method_args.n_components);
    switch (method_args.get_cost_func()) {
        case METHOD_UMAP:
            stderr_printf("UMAP embedding parameters a = %.3f, b = %.3f, gamma = %.3f\n", method_args.a, method_args.b,
                          method_args.gamma);
            break;
        case METHOD_LARGEVIZ:
            stderr_printf("LargeVis embedding parameters gamma = %.3f\n", method_args.gamma);
            break;
        case METHOD_LEOPOLD:
            stderr_printf("Leopold embedding parameters b = %.3f\n", method_args.b);
            break;
        case METHOD_LEOPOLD2:
            stderr_printf("Leopold2 embedding parameters b = %.3f\n", method_args.b);
            break;
    }
    stderr_printf("Optimizing for %d epochs with %d threads \n", method_args.n_epochs, (int)method_args.n_threads);
    if (method_args.debug_runtime_diagnostics) {
        stderr_printf("Runtime diagnostics: requested_threads=%zu, effective_threads=%zu, max_threads=%u\n",
                      requested_threads, method_args.n_threads, get_max_threads());
        stderr_printf("Runtime diagnostics: batch=%s, grain_size=%zu, approx_pow=%s, rng_type='%s', optimizer='%s'\n",
                      method_args.batch ? "true" : "false", method_args.grain_size,
                      method_args.approx_pow ? "true" : "false", method_args.get_rng_type().c_str(),
                      optimizer_name(method_args.opt_args));
        stderr_printf("Runtime diagnostics: OMP_NUM_THREADS=%s, OPENBLAS_NUM_THREADS=%s, MKL_NUM_THREADS=%s\n",
                      env_or_unset("OMP_NUM_THREADS"), env_or_unset("OPENBLAS_NUM_THREADS"),
                      env_or_unset("MKL_NUM_THREADS"));
        stderr_printf("Runtime diagnostics: SLURM_CPUS_PER_TASK=%s, NSLOTS=%s, OMP_PROC_BIND=%s, OMP_PLACES=%s\n",
                      env_or_unset("SLURM_CPUS_PER_TASK"), env_or_unset("NSLOTS"),
                      env_or_unset("OMP_PROC_BIND"), env_or_unset("OMP_PLACES"));
        const unsigned int affinity_cpus = affinity_cpu_count();
        if (affinity_cpus > 0) {
            stderr_printf("Runtime diagnostics: main-thread sched_getaffinity CPUs=%u\n", affinity_cpus);
        }
        else {
            stderr_printf("Runtime diagnostics: main-thread sched_getaffinity CPUs=<unavailable>\n");
        }
    }
    FLUSH;
};

void create_umap(UmapFactory& UF, UwotArgs& method_args) {
    if (method_args.approx_pow) {
        const uwot::apumap_gradient gradient(method_args.a, method_args.b, method_args.gamma);
        UF.create(gradient, method_args.get_engine());
    }
    else {
        const uwot::umap_gradient gradient(method_args.a, method_args.b, method_args.gamma);
        UF.create(gradient, method_args.get_engine());
    }
}

void create_tumap(UmapFactory& UF, UwotArgs& method_args) {
    const uwot::tumap_gradient gradient(method_args.gamma);
    UF.create(gradient, method_args.get_engine());
}

void create_largevis(UmapFactory& UF, UwotArgs& method_args) {
    const uwot::largevis_gradient gradient(method_args.gamma);
    UF.create(gradient, method_args.get_engine());
}

void create_umapai(UmapFactory& UF, UwotArgs& method_args) {
    const std::size_t ndim = static_cast<std::size_t>(method_args.n_components);
    const uwot::umapai_gradient gradient(method_args.ai, method_args.b, ndim);
    UF.create(gradient, method_args.get_engine());
}

void create_umapai2(UmapFactory& UF, UwotArgs& method_args) {
    const std::size_t ndim = static_cast<std::size_t>(method_args.n_components);
    const uwot::umapai2_gradient gradient(method_args.ai, method_args.aj, method_args.b, ndim);
    UF.create(gradient, method_args.get_engine());
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

EdgeVectors buildEdgeVectors(arma::sp_mat& G, const UwotArgs& uwot_args) {
    arma::sp_mat H = G;
    double w_max = arma::max(arma::max(H));
    H.clean(w_max / uwot_args.n_epochs);

    arma::sp_mat Ht = arma::trans(H).eval();

    unsigned int nV = G.n_rows;
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

    EdgeVectors EV;
    EV.positive_head = std::move(positive_head);
    EV.positive_tail = std::move(positive_tail);
    EV.positive_ptr = std::move(positive_ptr);
    EV.epochs_per_sample = std::move(epochs_per_sample);
    EV.n_vertices = nV;

    return EV;
}

void validateMethodArgs(const UwotArgs& uwot_args, std::size_t n_vertices) {
    if (uwot_args.get_cost_func() == METHOD_LEOPOLD) {
        if (uwot_args.ai.size() != n_vertices) {
            throw std::invalid_argument("For method='leopold', 'ai' must have length equal to number of vertices");
        }
    }
    else if (uwot_args.get_cost_func() == METHOD_LEOPOLD2) {
        if (uwot_args.ai.size() != n_vertices || uwot_args.aj.size() != n_vertices) {
            throw std::invalid_argument("For method='leopold2', 'ai' and 'aj' must have length equal to number of vertices");
        }
    }
}

} // anonymous namespace

namespace actionet {

arma::mat optimize_layout_uwot(arma::sp_mat& G, arma::mat& initial_coordinates, UwotArgs uwot_args) {
    if (G.n_cols != G.n_rows) {
        throw std::invalid_argument("'G' must be a square matrix");
    }

    if (G.n_cols != initial_coordinates.n_rows) {
        throw std::invalid_argument("Incompatible dimsensions (G.n_cols != initial_coordinates.n_rows)");
    }
    if (initial_coordinates.n_cols < uwot_args.n_components) {
        throw std::invalid_argument("'initial_coordinates' must have at least n_components columns");
    }

    const std::size_t requested_threads = uwot_args.n_threads;
    uwot_args.n_threads = get_num_threads(0, static_cast<int>(uwot_args.n_threads));
    if (uwot_args.n_epochs <= 0) {
        uwot_args.n_epochs = (initial_coordinates.n_rows <= 10000) ? 500 : 200; // uwot defaults
    }

    // `UF` references `coords`. Must be in the same scope.
    uwot::Coords coords = getCoords(initial_coordinates, uwot_args.n_components);
    auto edge_vectors = buildEdgeVectors(G, uwot_args);
    auto& positive_head = edge_vectors.positive_head;
    auto& positive_tail = edge_vectors.positive_tail;
    auto& epochs_per_sample = edge_vectors.epochs_per_sample;
    auto& positive_ptr = edge_vectors.positive_ptr;
    const unsigned int n_vertices = edge_vectors.n_vertices;
    validateMethodArgs(uwot_args, n_vertices);

    bool move_other = true;
    UmapFactory UF(move_other, uwot_args.get_rng_type(),
                   coords.get_head_embedding(), coords.get_tail_embedding(),
                   positive_head, positive_tail, positive_ptr, uwot_args.n_epochs,
                   n_vertices, n_vertices, epochs_per_sample, uwot_args.alpha,
                   uwot_args.opt_args, uwot_args.negative_sample_rate, uwot_args.batch,
                   uwot_args.n_threads, uwot_args.grain_size, uwot_args.verbose);

    if (uwot_args.verbose) { verboseStatus(uwot_args, requested_threads); }

    switch (uwot_args.get_cost_func()) {
        case METHOD_TUMAP:
            create_tumap(UF, uwot_args);
            break;
        case METHOD_LARGEVIZ:
            create_largevis(UF, uwot_args);
            break;
        case METHOD_LEOPOLD:
            create_umapai(UF, uwot_args);
            break;
        case METHOD_LEOPOLD2:
            create_umapai2(UF, uwot_args);
            break;
        case METHOD_UMAP:
        default:
            create_umap(UF, uwot_args);
    }

    arma::fmat uwot_embedding(UF.head_embedding.data(), uwot_args.n_components, G.n_rows);
    arma::mat coords_out = arma::trans(arma::conv_to<arma::mat>::from(uwot_embedding));
    stderr_printf("Optimization finished\n");
    FLUSH;

    return (coords_out);
}

} // namespace actionet
