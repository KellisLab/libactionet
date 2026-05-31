#include "visualization/uwot_actionet.hpp"
#include "visualization/UmapFactory.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "uwot/coords.h"
#include "uwot/connected_components.h"

#include <algorithm>
#include <cstdlib>
#include <random>
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
    // Cleaned graph (post H.clean) and its transpose, retained so the caller
    // can run connected-components / disconnected-vertex repair without
    // re-doing the prune.
    arma::sp_mat H;
    arma::sp_mat Ht;
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
    EV.H = std::move(H);
    EV.Ht = std::move(Ht);

    return EV;
}

// Compute connected components of the (undirected closure of the) post-pruned
// graph by calling the vendored `uwot::connected_components_undirected` helper.
// arma stores `sp_mat` in CSC; the helper expects a CSR/CSR-like (indices,
// indptr) pair for both the graph and its transpose. Since CSC of `H` equals
// CSR of `H^T`, and CSC of `Ht == H^T` equals CSR of `H`, we feed:
//   indices1, indptr1 = CSR of H   = (Ht.row_indices, Ht.col_ptrs)
//   indices2, indptr2 = CSR of H^T = (H.row_indices,  H.col_ptrs)
std::pair<unsigned int, std::vector<int>>
computeConnectedComponents(const arma::sp_mat& H, const arma::sp_mat& Ht) {
    const std::size_t n = H.n_rows;

    auto to_int_vec = [](const arma::uword* data, std::size_t count) {
        std::vector<int> out(count);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<int>(data[i]);
        }
        return out;
    };

    // CSR of H: row pointers come from Ht.col_ptrs (size n+1), column indices
    // from Ht.row_indices (size nnz).
    std::vector<int> indptr1 = to_int_vec(Ht.col_ptrs, Ht.n_cols + 1);
    std::vector<int> indices1 = to_int_vec(Ht.row_indices, Ht.n_nonzero);

    // CSR of H^T (i.e., row-pointers from H.col_ptrs).
    std::vector<int> indptr2 = to_int_vec(H.col_ptrs, H.n_cols + 1);
    std::vector<int> indices2 = to_int_vec(H.row_indices, H.n_nonzero);

    return uwot::connected_components_undirected(n, indices1, indptr1, indices2, indptr2);
}

// Canonical umap-learn safeguard for disconnected/orphaned vertices.
//
// Mirrors `umap.umap_.simplicial_set_embedding` behavior: if the post-pruned
// graph has more than one connected component, translate every vertex outside
// the largest component by a random offset drawn per-component from
// N(0, COMPONENT_OFFSET_SCALE), preserving the relative layout within each
// non-largest component while breaking axis-alignment of frozen seeds. A small
// per-coordinate Gaussian jitter (scale `JITTER_SCALE`) is then applied to the
// entire embedding, which also handles duplicate-row seeds within the largest
// component.
//
// The jitter step is intentionally always applied (when repair is enabled),
// even with a single connected component, because the diagnosed failure mode
// (cells with average archetype-1/2 footprints landing exactly at X=0 or
// Y=0 after `scale()`) is independent of graph connectivity.
//
// Reference: lmcinnes/umap simplicial_set_embedding `noisy_scale_coords` and
// the per-component recentering loop.
void applyDisconnectedRepair(arma::mat& initial_coordinates,
                             const arma::sp_mat& H,
                             const arma::sp_mat& Ht,
                             unsigned int n_components,
                             std::mt19937_64& engine,
                             bool verbose) {
    constexpr float COMPONENT_OFFSET_SCALE = 10.0f; // matches umap-learn default
    constexpr float JITTER_SCALE = 1e-4f;           // matches umap-learn default

    const std::size_t n_vertices = initial_coordinates.n_rows;
    const std::size_t ndim = static_cast<std::size_t>(n_components);

    if (n_vertices == 0 || ndim == 0) {
        return;
    }

    auto [n_comp, comp_labels] = computeConnectedComponents(H, Ht);

    if (n_comp > 1) {
        std::vector<std::size_t> comp_sizes(n_comp, 0);
        for (auto label : comp_labels) {
            ++comp_sizes[static_cast<std::size_t>(label)];
        }
        const std::size_t largest = static_cast<std::size_t>(
            std::distance(comp_sizes.begin(),
                          std::max_element(comp_sizes.begin(), comp_sizes.end())));

        // Per-component centroid (only for non-largest components, since the
        // largest is left untouched).
        arma::mat centroids(n_comp, ndim, arma::fill::zeros);
        for (std::size_t v = 0; v < n_vertices; ++v) {
            const std::size_t c = static_cast<std::size_t>(comp_labels[v]);
            if (c == largest) continue;
            for (std::size_t d = 0; d < ndim; ++d) {
                centroids(c, d) += initial_coordinates(v, d);
            }
        }
        for (std::size_t c = 0; c < n_comp; ++c) {
            if (c == largest) continue;
            const double denom = static_cast<double>(std::max<std::size_t>(1, comp_sizes[c]));
            for (std::size_t d = 0; d < ndim; ++d) {
                centroids(c, d) /= denom;
            }
        }

        // Random per-component offset (largest stays at origin offset).
        std::normal_distribution<float> offset_dist(0.0f, COMPONENT_OFFSET_SCALE);
        arma::mat offsets(n_comp, ndim, arma::fill::zeros);
        for (std::size_t c = 0; c < n_comp; ++c) {
            if (c == largest) continue;
            for (std::size_t d = 0; d < ndim; ++d) {
                offsets(c, d) = offset_dist(engine);
            }
        }

        // Translate non-largest components: subtract their centroid (so each
        // component's internal layout is preserved relative to its centroid)
        // and add the random offset.
        std::size_t n_relocated = 0;
        for (std::size_t v = 0; v < n_vertices; ++v) {
            const std::size_t c = static_cast<std::size_t>(comp_labels[v]);
            if (c == largest) continue;
            for (std::size_t d = 0; d < ndim; ++d) {
                initial_coordinates(v, d) -= centroids(c, d);
                initial_coordinates(v, d) += offsets(c, d);
            }
            ++n_relocated;
        }

        if (verbose) {
            stderr_printf(
                "Disconnected-vertex repair: %u connected components found; "
                "relocated %zu vertices outside the largest component (size %zu)\n",
                n_comp, n_relocated, comp_sizes[largest]);
        }
    }

    // Always apply small per-coordinate jitter (umap-learn `noisy_scale_coords`).
    // This breaks any remaining seed-collinearity for duplicate or zero rows
    // (e.g. cells whose archetype-footprint row standardizes to a vector of
    // axis-aligned values) within the largest component.
    std::normal_distribution<float> jitter_dist(0.0f, JITTER_SCALE);
    for (std::size_t v = 0; v < n_vertices; ++v) {
        for (std::size_t d = 0; d < ndim; ++d) {
            initial_coordinates(v, d) += jitter_dist(engine);
        }
    }
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

    // Build edge vectors first so we can run the canonical disconnected-vertex
    // repair on the post-pruned graph before seeding `getCoords`.
    auto edge_vectors = buildEdgeVectors(G, uwot_args);

    // Apply the umap-learn safeguard. We work on a local copy to avoid
    // mutating the caller's matrix (which may live in R/Python memory).
    // Only the first `n_components` columns of `initial_coordinates` are used
    // by `getCoords`, so we copy just those.
    arma::mat init_for_layout =
        initial_coordinates.cols(0, uwot_args.n_components - 1);
    if (uwot_args.repair_disconnected) {
        applyDisconnectedRepair(init_for_layout, edge_vectors.H, edge_vectors.Ht,
                                uwot_args.n_components, uwot_args.get_engine(),
                                uwot_args.verbose);
    }

    // `UF` references `coords`. Must be in the same scope.
    uwot::Coords coords = getCoords(init_for_layout, uwot_args.n_components);
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
