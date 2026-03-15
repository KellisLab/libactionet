// Internal core kNN builder: accepts raw row-major float32 input, returns CSRGraph.
// This is the performance-critical path used by the Python binding.
// The legacy actionet::buildNetwork(arma::mat) API is preserved as a shim over this core.
//
// Design constraints:
//   - No Armadillo dense matrix involved in the hot path.
//   - Input: row-major float32 pointer (n_points x dim).  Each row is one point.
//   - Output: CSRGraph — plain CSR buffers, no Armadillo dependency.
//   - Thread safety: thread_no=0 means auto-detect.
//   - The caller owns the input pointer; this function does not free it.
#ifndef ACTIONET_BUILD_NETWORK_CORE_HPP
#define ACTIONET_BUILD_NETWORK_CORE_HPP

#include "libactionet_config.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace actionet {

    using CSRVertexIndex = std::uint32_t;
    using CSROffset = std::uint64_t;

    /// Compact CSR graph with 64-bit-safe counts/offsets and float32 weights.
    /// Represents a symmetric cells x cells adjacency matrix.
    struct CSRGraph {
        CSROffset                     n = 0;      ///< Number of vertices (rows == cols).
        std::vector<CSROffset>        indptr;     ///< Row pointer array, length n+1.
        std::vector<CSRVertexIndex>   indices;    ///< Column index array, length nnz.
        std::vector<float>            data;       ///< Edge weight array, length nnz.

        CSROffset nnz() const { return static_cast<CSROffset>(data.size()); }
    };

    // Transitional alias for downstream callers still using the old name.
    using CSRGraph32 = CSRGraph;

    /// Parameters for the core kNN graph builder.
    /// Mirrors the public buildNetwork() parameter set exactly.
    struct BuildNetworkParams {
        std::string algorithm        = "k*nn"; ///< "k*nn" or "knn"
        std::string distance_metric  = "jsd";  ///< "jsd", "l2", or "ip"
        double      density          = 1.0;
        int         thread_no        = 0;
        double      M                = 16;
        /// For algorithm="k*nn": effective ef_construction is max(ef_construction, kNN)
        /// because the adaptive search radius grows to O(sqrt(N)) and a lower value
        /// would degrade recall at the required query sizes.  User-supplied values
        /// larger than kNN are respected, which can change graph topology relative
        /// to older releases that always forced ef_construction = kNN.
        double      ef_construction  = 200;
        /// For algorithm="k*nn": effective ef is max(ef, kNN) for the same reason.
        /// Larger user-supplied values are preserved instead of being discarded.
        double      ef               = 200;
        bool        mutual_edges_only = true;
        int         k                = 10;     ///< Used only for algorithm="knn"
    };

    /// @brief Core kNN graph builder.  Layout-neutral hot path.
    ///
    /// @param X       Row-major float32 input, shape (n_points x dim).
    ///                Each row is one data point.  Caller retains ownership.
    /// @param n_points Number of data points (rows in X).
    /// @param dim      Feature dimensionality (columns in X).
    /// @param params   Algorithm parameters.
    ///
    /// @return CSRGraph symmetric adjacency matrix (cells x cells).
    CSRGraph buildNetworkCore(const float*              X,
                              std::size_t               n_points,
                              std::size_t               dim,
                              const BuildNetworkParams& params);

    /// @brief Convert a CSRGraph to an arma::sp_mat (CSC, double).
    /// Used by the legacy armadillo shim and the R wrapper.
    arma::sp_mat armaSpMatFromCSR(const CSRGraph& g);

} // namespace actionet

#endif // ACTIONET_BUILD_NETWORK_CORE_HPP
