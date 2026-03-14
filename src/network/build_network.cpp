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

struct DirectedEdge {
    VertexIndex src;
    VertexIndex dst;
    float value;
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
// Internal helpers: build symmetric CSR from directed kNN edge lists
// ---------------------------------------------------------------------------

// Given directed edge triplets (src, dst, dist), apply distance-to-similarity
// transform, symmetrize with the legacy semantics, zero the diagonal, and
// return a CSRGraph.
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

    // --- distance -> similarity ---------------------------------------------
    if (distance_metric == "jsd") {
        for (std::size_t e = 0; e < nnz_dir; ++e) {
            dists[e] = std::max(epsilon, 1.0f - dists[e]);
        }
    }
    else {
        std::vector<float> max_d(n, 0.0f);
        for (std::size_t e = 0; e < nnz_dir; ++e) {
            const auto dst = static_cast<std::size_t>(dsts[e]);
            if (dists[e] > max_d[dst]) {
                max_d[dst] = dists[e];
            }
        }
        for (std::size_t e = 0; e < nnz_dir; ++e) {
            const auto dst = static_cast<std::size_t>(dsts[e]);
            dists[e] = std::max(epsilon, max_d[dst] - dists[e]);
        }
    }

    // Sort directed edges by unordered pair key (lo, hi) so that the forward
    // edge (lo→hi) and its reverse (hi→lo) are always adjacent in the sorted
    // order, regardless of how many other edges each endpoint has.
    // Sorting by (src, dst) alone would interleave them with edges to other
    // neighbours, so the inner accumulation loop below would never see both
    // directions together and mutual-only graphs would always be empty.
    std::vector<std::size_t> order(nnz_dir);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const auto lo_a = std::min(srcs[a], dsts[a]);
        const auto hi_a = std::max(srcs[a], dsts[a]);
        const auto lo_b = std::min(srcs[b], dsts[b]);
        const auto hi_b = std::max(srcs[b], dsts[b]);
        if (lo_a != lo_b) return lo_a < lo_b;
        if (hi_a != hi_b) return hi_a < hi_b;
        // Tie-break by directed (src, dst) so deduplication is stable.
        if (srcs[a] != srcs[b]) return srcs[a] < srcs[b];
        return dsts[a] < dsts[b];
    });

    std::vector<DirectedEdge> edges;
    edges.reserve(nnz_dir);
    for (const auto idx : order) {
        const DirectedEdge edge{srcs[idx], dsts[idx], dists[idx]};
        if (!edges.empty() && edges.back().src == edge.src && edges.back().dst == edge.dst) {
            edges.back().value += edge.value;
        }
        else {
            edges.push_back(edge);
        }
    }

    std::vector<std::vector<std::pair<VertexIndex, float>>> rows(n);

    for (std::size_t i = 0; i < edges.size();) {
        const auto u0 = edges[i].src;
        const auto v0 = edges[i].dst;
        if (u0 == v0) {
            ++i;
            continue;
        }

        const auto lo = std::min(u0, v0);
        const auto hi = std::max(u0, v0);

        double w_lo_hi = 0.0;
        double w_hi_lo = 0.0;

        while (i < edges.size()) {
            const auto& edge = edges[i];
            if (std::min(edge.src, edge.dst) != lo || std::max(edge.src, edge.dst) != hi) {
                break;
            }

            if (edge.src == lo && edge.dst == hi) {
                w_lo_hi += edge.value;
            }
            else if (edge.src == hi && edge.dst == lo) {
                w_hi_lo += edge.value;
            }
            ++i;
        }

        float w_sym = 0.0f;
        if (mutual_edges_only) {
            if (w_lo_hi <= 0.0 || w_hi_lo <= 0.0) {
                continue;
            }
            w_sym = std::sqrt(static_cast<float>(w_lo_hi * w_hi_lo));
        }
        else {
            const double combined = w_lo_hi + w_hi_lo;
            if (combined <= 0.0) {
                continue;
            }
            // Match the legacy (G + G.t()) / 2 behaviour exactly.
            w_sym = static_cast<float>(0.5 * combined);
        }

        rows[static_cast<std::size_t>(lo)].emplace_back(hi, w_sym);
        rows[static_cast<std::size_t>(hi)].emplace_back(lo, w_sym);
    }

    actionet::CSRGraph g;
    g.n = static_cast<CSROffset>(n);
    g.indptr.resize(n + 1, CSROffset{0});

    for (std::size_t row = 0; row < n; ++row) {
        g.indptr[row + 1] = static_cast<CSROffset>(rows[row].size());
    }
    for (std::size_t row = 0; row < n; ++row) {
        g.indptr[row + 1] += g.indptr[row];
    }

    const auto total_nnz = static_cast<std::size_t>(g.indptr[n]);
    g.indices.resize(total_nnz);
    g.data.resize(total_nnz);

    for (std::size_t row = 0; row < n; ++row) {
        auto& row_entries = rows[row];
        std::sort(row_entries.begin(), row_entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        const auto offset = static_cast<std::size_t>(g.indptr[row]);
        for (std::size_t j = 0; j < row_entries.size(); ++j) {
            g.indices[offset + j] = row_entries[j].first;
            g.data[offset + j] = row_entries[j].second;
        }
    }

    return g;
}

// ---------------------------------------------------------------------------
// Core builder: k*-Nearest Neighbors (adaptive k, NIPS 2016)
// ---------------------------------------------------------------------------
static actionet::CSRGraph
buildNetworkCore_KstarNN(const float*                     X,
                         std::size_t                      n,
                         std::size_t                      dim,
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
    if (n < 2) {
        return make_empty_graph(n);
    }

    const double LC = 1.0 / p.density;

    std::vector<float> X_norm_buf;
    const float* Xq = X;

    if (p.distance_metric == "jsd") {
        X_norm_buf.assign(X, X + checked_product(n, dim, "Normalized k*nn input"));
        for (std::size_t i = 0; i < n; ++i) {
            float* row = X_norm_buf.data() + i * dim;
            float sum = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                row[d] = std::max(0.0f, std::min(1.0f, row[d]));
                sum += row[d];
            }
            if (sum > 0.0f) {
                for (std::size_t d = 0; d < dim; ++d) {
                    row[d] /= sum;
                }
            }
        }
        Xq = X_norm_buf.data();
    }

    const auto kNN = compute_kstar_knn(n);
    const double ef_val = static_cast<double>(kNN);

    stdout_printf("\tBuilding index ... ");
    FLUSH;

    auto idx_kstar = makeHnswIndex(p.distance_metric, n, checked_hnsw_dim(dim), p.M, ef_val);
    idx_kstar.hnsw->setEf(ef_val);

    const int threads_use = checked_hnsw_threads(n, p.thread_no);
    const auto n_ll = static_cast<long long>(n);
    #pragma omp parallel for num_threads(threads_use)
    for (long long j = 0; j < n_ll; ++j) {
        const auto idx = static_cast<std::size_t>(j);
        idx_kstar.hnsw->addPoint(Xq + idx * dim, idx);
    }

    stdout_printf("done\n");
    FLUSH;
    stdout_printf("\tIdentifying nearest neighbors ... ");
    FLUSH;

    const auto knn_buffer_size = checked_product(n, kNN + 1, "k*nn neighbor buffers");
    std::vector<hnswlib::labeltype> idx_flat(knn_buffer_size);
    std::vector<float> dist_flat(knn_buffer_size);

    #pragma omp parallel for num_threads(threads_use)
    for (long long i = 0; i < n_ll; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        auto result = idx_kstar.hnsw->searchKnn(Xq + idx * dim, kNN + 1);

        auto* idx_row = idx_flat.data() + idx * (kNN + 1);
        auto* dist_row = dist_flat.data() + idx * (kNN + 1);

        for (std::size_t j = kNN + 1; j-- > 0;) {
            const auto& top = result.top();
            dist_row[j] = top.first;
            idx_row[j] = top.second;
            result.pop();
        }
    }
    // idx_kstar goes out of scope here — HierarchicalNSW and SpaceInterface are deleted.

    stdout_printf("done\n");
    FLUSH;
    stdout_printf("\tConstructing adaptive-nearest neighbor graph ... ");
    FLUSH;

    if (p.distance_metric == "jsd") {
        for (auto& d : dist_flat) {
            d = std::max(0.0f, std::min(1.0f, d));
        }
    }

    std::vector<float> lambda_flat(knn_buffer_size, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const float* dist_row = dist_flat.data() + i * (kNN + 1);
        float* lambda_row = lambda_flat.data() + i * (kNN + 1);
        double beta_sum = 0.0;
        double beta_sq_sum = 0.0;
        for (std::size_t k = 1; k <= kNN; ++k) {
            const double beta = LC * static_cast<double>(dist_row[k]);
            beta_sum += beta;
            beta_sq_sum += beta * beta;
            const double inner = static_cast<double>(k) + beta_sum * beta_sum
                - static_cast<double>(k) * beta_sq_sum;
            const double lambda = (1.0 / static_cast<double>(k))
                * (beta_sum + std::sqrt(std::max(0.0, inner)));
            lambda_row[k] = static_cast<float>(lambda);
        }
    }

    constexpr std::size_t MAX_RESERVE = 500000000ULL;
    const auto estimated = checked_product(n, kNN, "k*nn edge estimate");
    const auto reserve = std::min(estimated, MAX_RESERVE);

    std::vector<VertexIndex> all_srcs;
    std::vector<VertexIndex> all_dsts;
    std::vector<float> all_dists;
    if (estimated < MAX_RESERVE) {
        try {
            all_srcs.reserve(reserve);
            all_dsts.reserve(reserve);
            all_dists.reserve(reserve);
        } catch (...) {
        }
    }

    #pragma omp parallel num_threads(threads_use)
    {
        std::vector<VertexIndex> local_srcs;
        std::vector<VertexIndex> local_dsts;
        std::vector<float> local_dists;

        #pragma omp for nowait
        for (long long v = 0; v < n_ll; ++v) {
            const auto dst = static_cast<std::size_t>(v);
            const float* dist_row = dist_flat.data() + dst * (kNN + 1);
            const auto* idx_row = idx_flat.data() + dst * (kNN + 1);
            const float* lambda_row = lambda_flat.data() + dst * (kNN + 1);

            std::size_t neighbor_no = kNN;
            for (std::size_t k = 1; k <= kNN; ++k) {
                const double beta = LC * static_cast<double>(dist_row[k]);
                const double delta = static_cast<double>(lambda_row[k]) - beta;
                if (delta < 0.0) {
                    neighbor_no = k;
                    break;
                }
            }

            for (std::size_t i = 1; i < neighbor_no; ++i) {
                local_srcs.push_back(
                    checked_vertex_index(static_cast<std::size_t>(idx_row[i]), "k*nn neighbor label")
                );
                local_dsts.push_back(static_cast<VertexIndex>(dst));
                local_dists.push_back(dist_row[i]);
            }
        }

        #pragma omp critical
        {
            all_srcs.insert(all_srcs.end(), local_srcs.begin(), local_srcs.end());
            all_dsts.insert(all_dsts.end(), local_dsts.begin(), local_dsts.end());
            all_dists.insert(all_dists.end(), local_dists.begin(), local_dists.end());
        }
    }

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

// ---------------------------------------------------------------------------
// Core builder: fixed-k KNN
// ---------------------------------------------------------------------------
static actionet::CSRGraph
buildNetworkCore_KNN(const float*                     X,
                     std::size_t                      n,
                     std::size_t                      dim,
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

    std::vector<float> X_norm_buf;
    const float* Xq = X;

    if (p.distance_metric == "jsd") {
        X_norm_buf.assign(X, X + checked_product(n, dim, "Normalized knn input"));
        for (std::size_t i = 0; i < n; ++i) {
            float* row = X_norm_buf.data() + i * dim;
            float sum = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                row[d] = std::max(0.0f, std::min(1.0f, row[d]));
                sum += row[d];
            }
            if (sum > 0.0f) {
                for (std::size_t d = 0; d < dim; ++d) {
                    row[d] /= sum;
                }
            }
        }
        Xq = X_norm_buf.data();
    }

    stdout_printf("\tBuilding index ... ");
    FLUSH;

    auto idx_knn = makeHnswIndex(p.distance_metric, n, checked_hnsw_dim(dim), p.M, p.ef_construction);
    idx_knn.hnsw->setEf(p.ef);

    const int threads_use = checked_hnsw_threads(n, p.thread_no);
    const auto n_ll = static_cast<long long>(n);
    #pragma omp parallel for num_threads(threads_use)
    for (long long j = 0; j < n_ll; ++j) {
        const auto idx = static_cast<std::size_t>(j);
        idx_knn.hnsw->addPoint(Xq + idx * dim, idx);
    }

    stdout_printf("done\n");
    FLUSH;
    stdout_printf("\tConstructing kNN edges ... ");
    FLUSH;

    constexpr std::size_t MAX_RESERVE = 500000000ULL;
    const auto estimated = checked_product(n, static_cast<std::size_t>(p.k), "knn edge estimate");
    const auto reserve = std::min(estimated, MAX_RESERVE);

    std::vector<VertexIndex> all_srcs;
    std::vector<VertexIndex> all_dsts;
    std::vector<float> all_dists;
    if (estimated < MAX_RESERVE) {
        try {
            all_srcs.reserve(reserve);
            all_dsts.reserve(reserve);
            all_dists.reserve(reserve);
        } catch (...) {
        }
    }

    #pragma omp parallel num_threads(threads_use)
    {
        std::vector<VertexIndex> local_srcs;
        std::vector<VertexIndex> local_dsts;
        std::vector<float> local_dists;

        #pragma omp for nowait
        for (long long i = 0; i < n_ll; ++i) {
            const auto src = static_cast<std::size_t>(i);
            const auto query_k = std::min(n, static_cast<std::size_t>(p.k) + 1);
            auto result = idx_knn.hnsw->searchKnn(Xq + src * dim, query_k);

            std::vector<std::pair<float, hnswlib::labeltype>> neighbors;
            neighbors.reserve(result.size());
            while (!result.empty()) {
                neighbors.push_back(result.top());
                result.pop();
            }

            int added = 0;
            for (auto it = neighbors.rbegin(); it != neighbors.rend() && added < p.k; ++it) {
                if (static_cast<std::size_t>(it->second) == src) {
                    continue;
                }
                local_srcs.push_back(static_cast<VertexIndex>(src));
                local_dsts.push_back(
                    checked_vertex_index(static_cast<std::size_t>(it->second), "knn neighbor label")
                );
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
    // idx_knn goes out of scope here — HierarchicalNSW and SpaceInterface are deleted.

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
