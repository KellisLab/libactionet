#include "network/build_network.hpp"
#include "network/build_network_core.hpp"
#include "network/hnsw_imp.hpp"
#include "utils_internal/utils_parallel.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Validation sets (file-scope)
// ---------------------------------------------------------------------------
static const std::set<std::string> distance_metrics = {"jsd", "l2", "ip"};
static const std::set<std::string> nn_approaches    = {"k*nn", "knn"};

namespace {

using VertexIndex = actionet::CSRVertexIndex;
using CSROffset = actionet::CSROffset;

// Directed edge before symmetrization (reused in intermediate sort).
struct DirectedEdge {
    VertexIndex src;
    VertexIndex dst;
    float value;
};

// Aggregated symmetric pair produced by symmetrize_to_csr Pass 1.
struct SymPair {
    VertexIndex lo;
    VertexIndex hi;
    float       w_sym;
};

inline VertexIndex checked_vertex_index(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<VertexIndex>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds the supported HNSW vertex range");
    }
    return static_cast<VertexIndex>(value);
}

inline arma::uword checked_arma_uword(std::uint64_t value, const char* name) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<arma::uword>::max())) {
        throw std::runtime_error(
            std::string("Graph exceeds arma::sp_mat output range for ") + name
        );
    }
    return static_cast<arma::uword>(value);
}

inline int checked_hnsw_dim(std::size_t dim) {
    if (dim > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Feature dimensionality exceeds supported HNSW range");
    }
    return static_cast<int>(dim);
}

inline int checked_hnsw_threads(std::size_t n_points, int thread_no) {
    const auto capped = static_cast<unsigned int>(
        std::min<std::size_t>(n_points, static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
    );
    return static_cast<int>(get_num_threads(capped, static_cast<unsigned int>(thread_no)));
}

inline std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* name) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    if (lhs > (std::numeric_limits<std::size_t>::max() / rhs)) {
        throw std::runtime_error(std::string(name) + " exceeds addressable memory on this platform");
    }
    return lhs * rhs;
}

inline actionet::CSRGraph make_empty_graph(std::size_t n) {
    actionet::CSRGraph g;
    g.n = static_cast<CSROffset>(n);
    g.indptr.assign(n + 1, CSROffset{0});
    return g;
}

inline std::size_t compute_kstar_knn(std::size_t n) {
    if (n < 2) {
        return 0;
    }
    const auto heuristic = static_cast<std::size_t>(5.0 * std::round(std::sqrt(static_cast<double>(n))));
    return std::min(n - 1, heuristic);
}

// ---------------------------------------------------------------------------
// ContiguousFloat32Reader: satisfies the PointReader concept for a row-major
// float32 buffer.  For l2/ip, load_row() returns a direct pointer into X with
// no copy.  For jsd, it clamps and normalizes the row into a per-thread scratch
// buffer.  scratch must not be shared between threads.
// ---------------------------------------------------------------------------
class ContiguousFloat32Reader {
    const float* X_;
    std::size_t  n_;
    std::size_t  dim_;
    bool         jsd_;
public:
    ContiguousFloat32Reader(const float* X, std::size_t n, std::size_t dim, bool jsd)
        : X_(X), n_(n), dim_(dim), jsd_(jsd) {}

    std::size_t n_points() const { return n_; }
    std::size_t dim()      const { return dim_; }

    const float* load_row(std::size_t i, std::vector<float>& scratch) const {
        const float* row = X_ + i * dim_;
        if (!jsd_) {
            return row;
        }
        scratch.resize(dim_);
        float sum = 0.0f;
        for (std::size_t d = 0; d < dim_; ++d) {
            scratch[d] = std::max(0.0f, std::min(1.0f, row[d]));
            sum += scratch[d];
        }
        if (sum > 0.0f) {
            const float inv = 1.0f / sum;
            for (std::size_t d = 0; d < dim_; ++d) {
                scratch[d] *= inv;
            }
        }
        return scratch.data();
    }
};

// ---------------------------------------------------------------------------
// AdaptiveScratch: per-thread working memory for the k*nn query loop.
// One instance per thread; must not be shared.
// ---------------------------------------------------------------------------
struct AdaptiveScratch {
    // JSD row normalization scratch (or future backed-reader row buffer).
    // Written by ContiguousFloat32Reader::load_row(); must not be shared.
    std::vector<float> row_buf;
    // Neighbor heap drain buffer: ascending order after drain+reverse.
    std::vector<std::pair<float, hnswlib::labeltype>> knn_result;
    // Per-thread directed edge accumulator — merged after the parallel region.
    std::vector<VertexIndex> local_srcs;
    std::vector<VertexIndex> local_dsts;
    std::vector<float>       local_dists;
};

// ---------------------------------------------------------------------------
// symmetrize_to_csr: two-pass direct CSR builder.
//
// Given directed edge triplets (srcs, dsts, dists):
//   Pass 1 – distance→similarity conversion, sort by unordered pair key,
//             aggregate duplicate directed edges, compute per-pair symmetric
//             weight and row degree counts.
//   Pass 2 – prefix-sum degrees into indptr, write both symmetric directions
//             directly into indices/data via a write_pos cursor.
//
// Row indices within each CSR row are naturally sorted without a post-fill
// sort.  Proof: for a fixed lo, entries written to row lo arrive with hi
// non-decreasing (pairs sorted by (lo,hi)).  For a fixed hi, entries written
// to row hi arrive with lo non-decreasing — also non-decreasing column order.
// ---------------------------------------------------------------------------
static actionet::CSRGraph
symmetrize_to_csr(std::vector<VertexIndex>  srcs,
                  std::vector<VertexIndex>  dsts,
                  std::vector<float>        dists,
                  std::size_t               n,
                  const std::string&        distance_metric,
                  bool                      mutual_edges_only)
{
    const float epsilon = 1e-7f;
    const std::size_t nnz_dir = srcs.size();

    if (dsts.size() != nnz_dir || dists.size() != nnz_dir) {
        throw std::runtime_error("Directed edge buffers must be the same length");
    }
    if (nnz_dir == 0) {
        return make_empty_graph(n);
    }

    // -----------------------------------------------------------------------
    // Pass 1a: distance → directed similarity
    // -----------------------------------------------------------------------
    if (distance_metric == "jsd") {
        for (std::size_t e = 0; e < nnz_dir; ++e) {
            dists[e] = std::max(epsilon, 1.0f - dists[e]);
        }
    } else {
        std::vector<float> max_d(n, 0.0f);
        for (std::size_t e = 0; e < nnz_dir; ++e) {
            const auto dst = static_cast<std::size_t>(dsts[e]);
            if (dists[e] > max_d[dst]) max_d[dst] = dists[e];
        }
        for (std::size_t e = 0; e < nnz_dir; ++e) {
            const auto dst = static_cast<std::size_t>(dsts[e]);
            dists[e] = std::max(epsilon, max_d[dst] - dists[e]);
        }
    }

    // -----------------------------------------------------------------------
    // Pass 1b: sort directed edges by unordered pair key (lo, hi, src, dst).
    //
    // Sorting by (min,max) ensures forward (lo→hi) and reverse (hi→lo) are
    // adjacent so the accumulation loop sees both directions together.
    // The directed tie-break makes deduplication of exact duplicate edges
    // stable.
    // -----------------------------------------------------------------------
    std::vector<std::size_t> order(nnz_dir);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const VertexIndex lo_a = std::min(srcs[a], dsts[a]);
        const VertexIndex hi_a = std::max(srcs[a], dsts[a]);
        const VertexIndex lo_b = std::min(srcs[b], dsts[b]);
        const VertexIndex hi_b = std::max(srcs[b], dsts[b]);
        if (lo_a != lo_b) return lo_a < lo_b;
        if (hi_a != hi_b) return hi_a < hi_b;
        if (srcs[a] != srcs[b]) return srcs[a] < srcs[b];
        return dsts[a] < dsts[b];
    });

    // -----------------------------------------------------------------------
    // Pass 1c: aggregate consecutive duplicate directed edges.
    // -----------------------------------------------------------------------
    std::vector<DirectedEdge> edges;
    edges.reserve(nnz_dir);
    for (const auto idx : order) {
        const DirectedEdge e{srcs[idx], dsts[idx], dists[idx]};
        if (!edges.empty() &&
            edges.back().src == e.src &&
            edges.back().dst == e.dst) {
            edges.back().value += e.value;
        } else {
            edges.push_back(e);
        }
    }

    // -----------------------------------------------------------------------
    // Pass 1d: walk unordered pairs, compute symmetric weight, count degrees.
    // flat_pairs holds one entry per accepted (lo,hi) pair.
    // degree[v] counts the number of symmetric neighbours of vertex v.
    // -----------------------------------------------------------------------
    std::vector<SymPair>     flat_pairs;
    flat_pairs.reserve(edges.size() / 2 + 1);
    std::vector<std::size_t> degree(n, 0);

    for (std::size_t i = 0; i < edges.size();) {
        const VertexIndex u0 = edges[i].src;
        const VertexIndex v0 = edges[i].dst;
        if (u0 == v0) { ++i; continue; }

        const VertexIndex lo = std::min(u0, v0);
        const VertexIndex hi = std::max(u0, v0);

        double w_lo_hi = 0.0;
        double w_hi_lo = 0.0;

        while (i < edges.size()) {
            const DirectedEdge& e = edges[i];
            if (std::min(e.src, e.dst) != lo || std::max(e.src, e.dst) != hi) break;
            if (e.src == lo) w_lo_hi += e.value;
            else             w_hi_lo += e.value;
            ++i;
        }

        float w_sym = 0.0f;
        if (mutual_edges_only) {
            if (w_lo_hi <= 0.0 || w_hi_lo <= 0.0) continue;
            w_sym = std::sqrt(static_cast<float>(w_lo_hi * w_hi_lo));
        } else {
            const double combined = w_lo_hi + w_hi_lo;
            if (combined <= 0.0) continue;
            // Arithmetic mean matches the legacy (G + G.t()) / 2 semantics.
            w_sym = static_cast<float>(0.5 * combined);
        }

        flat_pairs.push_back({lo, hi, w_sym});
        degree[static_cast<std::size_t>(lo)] += 1;
        degree[static_cast<std::size_t>(hi)] += 1;
    }

    // -----------------------------------------------------------------------
    // Pass 2a: prefix-sum degrees into indptr.
    // -----------------------------------------------------------------------
    actionet::CSRGraph g;
    g.n = static_cast<CSROffset>(n);
    g.indptr.resize(n + 1, CSROffset{0});
    for (std::size_t v = 0; v < n; ++v) {
        g.indptr[v + 1] = g.indptr[v] + static_cast<CSROffset>(degree[v]);
    }

    const auto total_nnz = static_cast<std::size_t>(g.indptr[n]);
    g.indices.resize(total_nnz);
    g.data.resize(total_nnz);

    // -----------------------------------------------------------------------
    // Pass 2b: write both symmetric directions via a write_pos cursor.
    // write_pos[v] starts at indptr[v] and advances as entries are written.
    // Row indices arrive in sorted column order (see proof in comment above).
    // -----------------------------------------------------------------------
    std::vector<std::size_t> write_pos(n);
    for (std::size_t v = 0; v < n; ++v) {
        write_pos[v] = static_cast<std::size_t>(g.indptr[v]);
    }

    for (const SymPair& sp : flat_pairs) {
        const std::size_t lo = static_cast<std::size_t>(sp.lo);
        const std::size_t hi = static_cast<std::size_t>(sp.hi);
        g.indices[write_pos[lo]] = sp.hi;
        g.data[write_pos[lo]]    = sp.w_sym;
        ++write_pos[lo];
        g.indices[write_pos[hi]] = sp.lo;
        g.data[write_pos[hi]]    = sp.w_sym;
        ++write_pos[hi];
    }

    return g;
}

// ---------------------------------------------------------------------------
// Core builder: k*-Nearest Neighbors (adaptive k, NIPS 2016)
//
// Refactored to use ContiguousFloat32Reader and per-thread AdaptiveScratch:
//   - No global X_norm_buf: JSD normalization happens per row in load_row().
//   - No global idx_flat / dist_flat / lambda_flat: eliminated entirely.
//   - Adaptive cutoff computed incrementally (no lambda array).
//   - Per-thread local edge accumulators; merged after the parallel region.
//   - ef and ef_construction floored at kNN (see BuildNetworkParams docs).
//   - Self filtering is label-based, which fixes duplicate-row / self-slot
//     correctness issues and can intentionally change graphs versus older code.
// ---------------------------------------------------------------------------
static actionet::CSRGraph
buildNetworkCore_KstarNN(const float*                        X,
                         std::size_t                         n,
                         std::size_t                         dim,
                         const actionet::BuildNetworkParams& p)
{
    stdout_printf("Building adaptive network (density = %.2f)\n", p.density);
    FLUSH;

    if (p.density <= 0.0) {
        throw std::runtime_error("density must be positive for k*nn");
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<VertexIndex>::max())) {
        throw std::runtime_error("Point count exceeds the supported HNSW vertex range");
    }
    // kNN = 0 for n < 2: return an empty graph immediately.
    if (n < 2) {
        return make_empty_graph(n);
    }

    const double LC     = 1.0 / p.density;
    const auto   kNN    = compute_kstar_knn(n);
    // ef and ef_construction are floored at kNN: the adaptive search radius
    // grows to O(sqrt(N)) and a smaller value would degrade recall.
    // Larger user-supplied values are also respected; this is intentional and
    // can change graph topology relative to older releases that always forced
    // both values to kNN.
    const double ef_c   = std::max(p.ef_construction, static_cast<double>(kNN));
    const double ef_q   = std::max(p.ef,               static_cast<double>(kNN));

    const bool   jsd    = (p.distance_metric == "jsd");
    const int    threads_use = checked_hnsw_threads(n, p.thread_no);
    const auto   n_ll   = static_cast<long long>(n);

    ContiguousFloat32Reader reader(X, n, dim, jsd);

    // -----------------------------------------------------------------------
    // Build HNSW index: rows fed through reader so JSD normalization happens
    // on demand without a full-matrix copy.
    // -----------------------------------------------------------------------
    stdout_printf("\tBuilding index ... ");
    FLUSH;

    auto idx_kstar = makeHnswIndex(p.distance_metric, n, checked_hnsw_dim(dim), p.M, ef_c);
    idx_kstar.hnsw->setEf(ef_q);

    // Each thread uses its own row_buf scratch for load_row().
    #pragma omp parallel num_threads(threads_use)
    {
        std::vector<float> row_buf;
        #pragma omp for nowait schedule(static)
        for (long long j = 0; j < n_ll; ++j) {
            const auto idx = static_cast<std::size_t>(j);
            const float* row = reader.load_row(idx, row_buf);
            idx_kstar.hnsw->addPoint(row, idx);
        }
    }

    stdout_printf("done\n");
    FLUSH;
    stdout_printf("\tConstructing adaptive-nearest neighbor graph ... ");
    FLUSH;

    // -----------------------------------------------------------------------
    // Query: per-thread AdaptiveScratch, no global neighbor buffers.
    // searchKnn returns a max-heap (farthest first); drain into a local
    // vector and iterate in reverse for ascending-distance order.
    //
    // Self-exclusion: skip by label — do not assume self is at a fixed slot.
    // This fixes duplicated-row and self-slot correctness issues.
    //
    // Adaptive cutoff: computed incrementally without a lambda array.
    // Emit neighbors 1..neighbor_no-1 (exclusive upper bound) after label-
    // based self filtering.  neighbor_no is set to position k at the first
    // failure (lambda < beta), so k itself is not emitted.
    // -----------------------------------------------------------------------
    std::vector<AdaptiveScratch> per_thread(static_cast<std::size_t>(threads_use));
    std::atomic<int> tid_counter{0};

    #pragma omp parallel num_threads(threads_use)
    {
        const int tid = tid_counter.fetch_add(1, std::memory_order_relaxed);
        AdaptiveScratch& sc = per_thread[static_cast<std::size_t>(tid)];

        #pragma omp for nowait schedule(static)
        for (long long v = 0; v < n_ll; ++v) {
            const auto src = static_cast<std::size_t>(v);
            const float* row = reader.load_row(src, sc.row_buf);

            // Drain the max-heap into sc.knn_result (farthest-first order),
            // then iterate in reverse for ascending-distance order.
            auto heap = idx_kstar.hnsw->searchKnn(row, kNN + 1);
            sc.knn_result.clear();
            sc.knn_result.reserve(heap.size());
            while (!heap.empty()) {
                sc.knn_result.push_back(heap.top());
                heap.pop();
            }
            // sc.knn_result is now farthest-first; iterate reversed = closest-first.

            // For JSD, clamp returned distances to [0,1] before the cutoff.
            if (jsd) {
                for (auto& pr : sc.knn_result) {
                    pr.first = std::max(0.0f, std::min(1.0f, pr.first));
                }
            }

            // Walk in ascending-distance order; skip self by label.
            std::size_t neighbor_no = kNN;  // default: keep all non-self neighbors
            double      beta_sum    = 0.0;
            double      beta_sq_sum = 0.0;
            std::size_t k           = 0;    // count of non-self neighbors seen

            for (auto it = sc.knn_result.rbegin(); it != sc.knn_result.rend(); ++it) {
                if (static_cast<std::size_t>(it->second) == src) continue;
                ++k;
                const double beta = LC * static_cast<double>(it->first);
                beta_sum    += beta;
                beta_sq_sum += beta * beta;
                const double inner  = static_cast<double>(k)
                    + beta_sum * beta_sum
                    - static_cast<double>(k) * beta_sq_sum;
                const double lambda = (1.0 / static_cast<double>(k))
                    * (beta_sum + std::sqrt(std::max(0.0, inner)));
                if (lambda < beta) {
                    // First failing neighbor: set cutoff to k (exclusive).
                    // Do not emit this neighbor.
                    neighbor_no = k;
                    break;
                }
                if (k >= kNN) break;
            }

            // Emit accepted neighbors (positions 1..neighbor_no-1 in
            // 1-indexed non-self order) into local edge accumulators.
            std::size_t emitted = 0;
            for (auto it = sc.knn_result.rbegin();
                 it != sc.knn_result.rend() && emitted < neighbor_no - 1;
                 ++it) {
                if (static_cast<std::size_t>(it->second) == src) continue;
                sc.local_srcs.push_back(
                    checked_vertex_index(static_cast<std::size_t>(it->second),
                                         "k*nn neighbor label"));
                sc.local_dsts.push_back(static_cast<VertexIndex>(src));
                sc.local_dists.push_back(it->first);
                ++emitted;
            }
        }
    }
    // idx_kstar goes out of scope here — HierarchicalNSW and SpaceInterface deleted.

    stdout_printf("done\n");
    FLUSH;
    stdout_printf("\tFinalizing network ... ");
    FLUSH;

    // -----------------------------------------------------------------------
    // Merge per-thread edge accumulators into flat arrays in one pass.
    // -----------------------------------------------------------------------
    std::size_t total_edges = 0;
    for (const auto& sc : per_thread) {
        total_edges += sc.local_srcs.size();
    }

    std::vector<VertexIndex> all_srcs;
    std::vector<VertexIndex> all_dsts;
    std::vector<float>       all_dists;
    all_srcs.reserve(total_edges);
    all_dsts.reserve(total_edges);
    all_dists.reserve(total_edges);

    for (auto& sc : per_thread) {
        all_srcs.insert(all_srcs.end(), sc.local_srcs.begin(), sc.local_srcs.end());
        all_dsts.insert(all_dsts.end(), sc.local_dsts.begin(), sc.local_dsts.end());
        all_dists.insert(all_dists.end(), sc.local_dists.begin(), sc.local_dists.end());
        // Release per-thread memory immediately.
        sc.local_srcs.clear();  sc.local_srcs.shrink_to_fit();
        sc.local_dsts.clear();  sc.local_dsts.shrink_to_fit();
        sc.local_dists.clear(); sc.local_dists.shrink_to_fit();
    }

    auto g = symmetrize_to_csr(
        std::move(all_srcs),
        std::move(all_dsts),
        std::move(all_dists),
        n,
        p.distance_metric,
        p.mutual_edges_only
    );
    stdout_printf("done\n");
    FLUSH;
    return g;
}

// ---------------------------------------------------------------------------
// Core builder: fixed-k KNN
//
// Also uses ContiguousFloat32Reader for JSD normalization on demand,
// eliminating the X_norm_buf whole-matrix copy.
// ---------------------------------------------------------------------------
static actionet::CSRGraph
buildNetworkCore_KNN(const float*                        X,
                     std::size_t                         n,
                     std::size_t                         dim,
                     const actionet::BuildNetworkParams& p)
{
    stdout_printf("Building fixed-degree network (k = %d)\n", p.k);
    FLUSH;

    if (p.k < 0) {
        throw std::runtime_error("k must be non-negative");
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<VertexIndex>::max())) {
        throw std::runtime_error("Point count exceeds the supported HNSW vertex range");
    }
    if (n < 2 || p.k == 0) {
        return make_empty_graph(n);
    }

    const bool jsd         = (p.distance_metric == "jsd");
    const int  threads_use = checked_hnsw_threads(n, p.thread_no);
    const auto n_ll        = static_cast<long long>(n);

    ContiguousFloat32Reader reader(X, n, dim, jsd);

    stdout_printf("\tBuilding index ... ");
    FLUSH;

    auto idx_knn = makeHnswIndex(p.distance_metric, n, checked_hnsw_dim(dim),
                                  p.M, p.ef_construction);
    idx_knn.hnsw->setEf(p.ef);

    #pragma omp parallel num_threads(threads_use)
    {
        std::vector<float> row_buf;
        #pragma omp for nowait schedule(static)
        for (long long j = 0; j < n_ll; ++j) {
            const auto idx = static_cast<std::size_t>(j);
            idx_knn.hnsw->addPoint(reader.load_row(idx, row_buf), idx);
        }
    }

    stdout_printf("done\n");
    FLUSH;
    stdout_printf("\tConstructing kNN edges ... ");
    FLUSH;

    std::vector<VertexIndex> all_srcs;
    std::vector<VertexIndex> all_dsts;
    std::vector<float>       all_dists;

    constexpr std::size_t MAX_RESERVE = 500000000ULL;
    const auto estimated = checked_product(n, static_cast<std::size_t>(p.k), "knn edge estimate");
    if (estimated < MAX_RESERVE) {
        try {
            all_srcs.reserve(estimated);
            all_dsts.reserve(estimated);
            all_dists.reserve(estimated);
        } catch (...) {}
    }

    #pragma omp parallel num_threads(threads_use)
    {
        std::vector<float>  row_buf;
        std::vector<VertexIndex> local_srcs;
        std::vector<VertexIndex> local_dsts;
        std::vector<float>       local_dists;
        std::vector<std::pair<float, hnswlib::labeltype>> nbrs;

        #pragma omp for nowait schedule(static)
        for (long long i = 0; i < n_ll; ++i) {
            const auto src     = static_cast<std::size_t>(i);
            const auto query_k = std::min(n, static_cast<std::size_t>(p.k) + 1);
            const float* row   = reader.load_row(src, row_buf);

            auto heap = idx_knn.hnsw->searchKnn(row, query_k);
            nbrs.clear();
            nbrs.reserve(heap.size());
            while (!heap.empty()) {
                nbrs.push_back(heap.top());
                heap.pop();
            }
            // nbrs is farthest-first; iterate reversed for closest-first.
            // Self-exclusion by label (not position).
            int added = 0;
            for (auto it = nbrs.rbegin(); it != nbrs.rend() && added < p.k; ++it) {
                if (static_cast<std::size_t>(it->second) == src) continue;
                local_srcs.push_back(static_cast<VertexIndex>(src));
                local_dsts.push_back(
                    checked_vertex_index(static_cast<std::size_t>(it->second),
                                         "knn neighbor label"));
                local_dists.push_back(it->first);
                ++added;
            }
        }

        #pragma omp critical
        {
            all_srcs.insert(all_srcs.end(), local_srcs.begin(), local_srcs.end());
            all_dsts.insert(all_dsts.end(), local_dsts.begin(), local_dsts.end());
            all_dists.insert(all_dists.end(), local_dists.begin(), local_dists.end());
        }
    }
    // idx_knn goes out of scope here — HierarchicalNSW and SpaceInterface deleted.

    stdout_printf("done\n");
    FLUSH;
    stdout_printf("\tFinalizing network ... ");
    FLUSH;

    auto g = symmetrize_to_csr(
        std::move(all_srcs),
        std::move(all_dsts),
        std::move(all_dists),
        n,
        p.distance_metric,
        p.mutual_edges_only
    );
    stdout_printf("done\n");
    FLUSH;
    return g;
}

} // namespace

// ---------------------------------------------------------------------------
// namespace actionet: public API
// ---------------------------------------------------------------------------

namespace actionet {

CSRGraph buildNetworkCore(const float*              X,
                          std::size_t               n_points,
                          std::size_t               dim,
                          const BuildNetworkParams& params)
{
    if (distance_metrics.find(params.distance_metric) == distance_metrics.end()) {
        throw distMetException;
    }
    if (nn_approaches.find(params.algorithm) == nn_approaches.end()) {
        throw nnApproachException;
    }
    if (params.k < 0) {
        throw std::runtime_error("k must be non-negative");
    }
    if (dim == 0 && n_points > 0) {
        throw std::runtime_error("buildNetworkCore requires a positive feature dimension");
    }
    if (X == nullptr && n_points > 0) {
        throw std::runtime_error("buildNetworkCore received a null data pointer");
    }

    if (params.algorithm == "k*nn") {
        return buildNetworkCore_KstarNN(X, n_points, dim, params);
    }
    return buildNetworkCore_KNN(X, n_points, dim, params);
}

arma::sp_mat armaSpMatFromCSR(const CSRGraph& g)
{
    if (g.indptr.size() != static_cast<std::size_t>(g.n) + 1) {
        throw std::runtime_error("CSR indptr length does not match row count");
    }
    if (g.indices.size() != g.data.size()) {
        throw std::runtime_error("CSR indices/data lengths do not match");
    }

    const arma::uword n = checked_arma_uword(g.n, "row count");
    const arma::uword nnz = checked_arma_uword(g.nnz(), "nnz");

    arma::umat locations(2, nnz);
    arma::vec values(nnz);

    arma::uword idx = 0;
    for (std::size_t row = 0; row < static_cast<std::size_t>(g.n); ++row) {
        if (g.indptr[row + 1] < g.indptr[row]) {
            throw std::runtime_error("CSR indptr must be non-decreasing");
        }
        for (auto j = g.indptr[row]; j < g.indptr[row + 1]; ++j) {
            const auto edge_idx = static_cast<std::size_t>(j);
            if (edge_idx >= g.indices.size()) {
                throw std::runtime_error("CSR indptr points past the edge buffer");
            }
            locations(0, idx) = checked_arma_uword(row, "row index");
            locations(1, idx) = checked_arma_uword(g.indices[edge_idx], "column index");
            values(idx) = static_cast<double>(g.data[edge_idx]);
            ++idx;
        }
    }

    return arma::sp_mat(locations, values, n, n);
}

arma::sp_mat
buildNetwork(const arma::mat& H, std::string algorithm, std::string distance_metric,
             double density, int thread_no, double M, double ef_construction,
             double ef, bool mutual_edges_only, int k)
{
    if (distance_metrics.find(distance_metric) == distance_metrics.end()) {
        throw distMetException;
    }
    if (nn_approaches.find(algorithm) == nn_approaches.end()) {
        throw nnApproachException;
    }

    const std::size_t dim = H.n_rows;
    const std::size_t n_points = H.n_cols;

    std::vector<float> X(checked_product(n_points, dim, "Legacy buildNetwork input"));
    for (std::size_t col = 0; col < n_points; ++col) {
        const double* src = H.colptr(col);
        float* dst = X.data() + col * dim;
        for (std::size_t d = 0; d < dim; ++d) {
            dst[d] = static_cast<float>(src[d]);
        }
    }

    BuildNetworkParams params;
    params.algorithm = std::move(algorithm);
    params.distance_metric = std::move(distance_metric);
    params.density = density;
    params.thread_no = thread_no;
    params.M = M;
    params.ef_construction = ef_construction;
    params.ef = ef;
    params.mutual_edges_only = mutual_edges_only;
    params.k = k;

    const auto g = buildNetworkCore(X.data(), n_points, dim, params);
    return armaSpMatFromCSR(g);
}

} // namespace actionet
