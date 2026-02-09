// Successive projection algorithm (SPA)
#ifndef ACTIONET_SPA_HPP
#define ACTIONET_SPA_HPP

#include "libactionet_config.hpp"

// Exported
namespace actionet {
    /// @brief Stores the output of <code>runSPA()</code>.
    ///
    /// Members:
    /// - <b>selected_cols</b>: Indices of selected columns (candidate vertices).
    /// - <b>column_norms</b>: Norms of candidate columns.
    struct ResSPA {
        arma::uvec selected_cols;
        arma::vec column_norms;
    };

    /// @brief Run successive projections algorithm (SPA) for separable NMF.
    ///
    /// @param A Input matrix.
    /// @param k Number of candidate vertices to select.
    ///
    /// @return <code>ResSPA</code> with selected indices and norms.
    ResSPA runSPA(arma::mat& A, int k);
} // namespace actionet

#endif //ACTIONET_SPA_HPP
