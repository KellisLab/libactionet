#ifndef ACTIONET_LIBACTIONET_HPP
#define ACTIONET_LIBACTIONET_HPP

// Set variables, macros, and headers for package compilation and linking.
// #include "libactionet_config.hpp"

// Exported modules: Symbols defined in these headers are accessible through
// the `actionet` namespace and visible to language bindings.

// decomposition: Algorithms for matrix decomposition
#include "decomposition/matrix_operator.hpp"
#include "decomposition/svd_main.hpp"
#include "decomposition/orthogonalization.hpp"

// io: Backed HDF5 matrix operators
#ifndef LIBACTIONET_NO_HDF5
#include "io/backed_h5ad/create_backed_operator.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"
#include "io/backed_h5ad/h5ad_matrix_io.hpp"
#endif

// action: Main archetypal analysis for cell type identification (ACTION) module
#include "action/aa.hpp"
#include "action/action_decomp.hpp"
#include "action/action_main.hpp"
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
#include "visualization/layout_network.hpp"
#include "visualization/color_map.hpp"

// tools: Exported tools and convenience functions
#include "tools/autocorrelation.hpp"
#include "tools/enrichment.hpp"
#include "tools/guide_calling.hpp"
#include "tools/matrix_aggregate.hpp"
#include "tools/matrix_transform.hpp"
#include "tools/mwm.hpp"
#include "tools/xicor.hpp"

#endif //ACTIONET_LIBACTIONET_HPP
