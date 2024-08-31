#ifndef ACTIONET_LIBACTIONET_HPP
#define ACTIONET_LIBACTIONET_HPP

// Set variables, macros, and headers for package compilation and linking.
// #include "libactionet_config.hpp"

// Exported modules: Symbols defined in these headers are accessible through
// the `action` namespace and visible to interfaces.

// decomposition: Algorithms for matrix decomposition
#include "decomposition/svd_main.hpp"
#include "decomposition/orthogonalization.hpp"

// action: Main archetypal analysis for cell type identification (ACTION) module
#include "action/aa.hpp"
#include "action/action_decomp.hpp"
#include "action/action_post.hpp"
#include "action/reduce_kernel.hpp"
#include "action/simplex_regression.hpp"
#include "action/spa.hpp"

// network: Network construction and manipulation
#include "network/build_network.hpp"
#include "network/network_diffusion.hpp"
#include "network/label_propagation.hpp"
#include "network/network_measures.hpp"

// Network node annotation
#include "annotation/marker_stats.hpp"
#include "annotation/specificity.hpp"

// visualization: Generate embeddings for visualization
#include "visualization/generate_layout.hpp"

// tools: Exported tools and convenience functions
#include "tools/normalization.hpp"
#include "tools/autocorrelation.hpp"
#include "tools/mwm.hpp"
#include "tools/xicor.hpp"
#include "tools/enrichment.hpp"
#include "tools/matrix_misc.hpp"

#endif //ACTIONET_LIBACTIONET_HPP
