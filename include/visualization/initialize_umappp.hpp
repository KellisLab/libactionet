#ifndef MY_UMAPPP_INITIALIZE_HPP
#define MY_UMAPPP_INITIALIZE_HPP

#include "umappp/Options.hpp"
#include "umappp/Status.hpp"

#include "umappp/NeighborList.hpp"
#include "umappp/combine_neighbor_sets.hpp"
#include "umappp/find_ab.hpp"
#include "umappp/neighbor_similarities.hpp"
//#include "spectral_init.hpp"
#include "umappp/Status.hpp"


#include <random>
#include <cstdint>

/**
 * @file initialize.hpp
 * @brief Initialize the UMAP algorithm.
 */

namespace umappp {
    /**
     * @cond
     */
    namespace internal {
        inline int choose_num_epochs(int num_epochs, size_t size) {
            if (num_epochs < 0) {
                // Choosing the number of epochs. We use a simple formula to decrease
                // the number of epochs with increasing size, with the aim being that
                // the 'extra work' beyond the minimal 200 epochs should be the same
                // regardless of the numbe of observations. Given one calculation per
                // observation per epoch, this amounts to 300 * 10000 calculations at
                // the lower bound, so we simply choose a number of epochs that
                // equalizes the number of calculations for any number of observations.
                if (num_epochs < 0) {
                    constexpr int limit = 10000, minimal = 200, maximal = 300;
                    if (size <= limit) {
                        num_epochs = minimal + maximal;
                    }
                    else {
                        num_epochs = minimal + static_cast<int>(std::ceil(maximal * limit / static_cast<double>(size)));
                    }
                }
            }
            return num_epochs;
        }
    }

    /**
     * @endcond
     */

    /**
     * @tparam Index_ Integer type of the neighbor indices.
     * @tparam Float_ Floating-point type for the distances.
     *
     * @param x Indices and distances to the nearest neighbors for each observation.
     * Note the expectations in the `NeighborList` documentation.
     * @param num_dim Number of dimensions of the embedding.
     * @param[in, out] embedding Pointer to an array in which to store the embedding, where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
     * This is only used as input if `Options::init == InitializeMethod::NONE`, otherwise it is only used as output.
     * The lifetime of the array should be no shorter than the final call to `Status::run()`.
     * @param options Further options.
     * Note that `Options::num_neighbors` is ignored here.
     *
     * @return A `Status` object containing the initial state of the UMAP algorithm.
     * Further calls to `Status::run()` will update the embeddings in `embedding`.
     */
    template <typename Index_, typename Float_>
    Status<Index_, Float_> initialize_custom(NeighborList<Index_, Float_> x, int num_dim, Float_* embedding,
                                             Options options) {
        internal::NeighborSimilaritiesOptions<Float_> nsopt;
        nsopt.local_connectivity = options.local_connectivity;
        nsopt.bandwidth = options.bandwidth;
        nsopt.num_threads = options.num_threads;
        internal::neighbor_similarities(x, nsopt);

        internal::combine_neighbor_sets(x, static_cast<Float_>(options.mix_ratio));

        //    // Choosing the manner of initialization.
        //    if (options.initialize == InitializeMethod::SPECTRAL || options.initialize == InitializeMethod::SPECTRAL_ONLY) {
        //        bool attempt = internal::spectral_init(x, num_dim, embedding, options.num_threads);
        //        if (!attempt && options.initialize == InitializeMethod::SPECTRAL) {
        //            internal::random_init(x.size(), num_dim, embedding);
        //        }
        //    } else if (options.initialize == InitializeMethod::RANDOM) {
        //        internal::random_init(x.size(), num_dim, embedding);
        //    }

        // Finding a good a/b pair.
        if (options.a <= 0 || options.b <= 0) {
            auto found = internal::find_ab(options.spread, options.min_dist);
            options.a = found.first;
            options.b = found.second;
        }

        options.num_epochs = internal::choose_num_epochs(options.num_epochs, x.size());

        return Status<Index_, Float_>(
            internal::similarities_to_epochs<Index_, Float_>(x, options.num_epochs, options.negative_sample_rate),
            options,
            num_dim,
            embedding
        );
    }
}
#endif
