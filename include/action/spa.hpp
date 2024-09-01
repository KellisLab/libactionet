// Successive projection algorithm (SPA)
#ifndef ACTIONET_SPA_HPP
#define ACTIONET_SPA_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    // Structs
    /// @brief Stores the output of <code>run_SPA()</code>
    ///
    /// Contains the following members:
    /// - <b>selected_cols</b>: Indices of columns of <b>A</b> representing candidate vertices.
    /// - <b>column_norms</b>: Norms of candidate column vectors.
    struct ResSPA {
        arma::uvec selected_cols;
        arma::vec column_norms;
    };

    // Functions
    // Solves separable NMF problem
    /// @brief Run successive projections algorithm (SPA) to solve separable NMF
    ///
    /// @param A Input matrix.
    /// @param k Number of candidate vertices to solve for.
    ///
    /// @return <code>struct</code> of type <code>ResSPA</code>.
    ///
    /// @remark See <code>ResSPA</code>.
    ResSPA run_SPA(arma::mat& A, int k);
} // namespace actionet

#endif //ACTIONET_SPA_HPP
