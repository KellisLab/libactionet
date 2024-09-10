// Modified variant of required UmapFactory struct implemented in uwot Rcpp interface.
// Calls and controls uwot
// Key modifications:
//      Eliminated R-dependency.
//      Remove support for R callback.
//      Added shared RNG engine for reproducibility in multithreaded operation.

#ifndef ACTIONET_UMAPFACTORY_HPP
#define ACTIONET_UMAPFACTORY_HPP

#include "libactionet_config.hpp"
#include "uwot/epoch.h"
#include "uwot/optimize.h"
#include "uwot/sampler.h"

#include "uwot/rng.h"
#include "uwot/rparallel.h"
#include "OptimizerArgs.hpp"

// Template class specialization to handle different rng/batch combinations
template <bool DoBatch = true>
struct BatchRngFactory {
    using PcgFactoryType = batch_pcg_factory;
    using TauFactoryType = batch_tau_factory;
};

template <>
struct BatchRngFactory<false> {
    using PcgFactoryType = pcg_factory;
    using TauFactoryType = tau_factory;
};

struct UmapFactory {
    bool move_other;
    bool pcg_rand;
    std::vector<float>& head_embedding; // Must remain reference (input and output)
    std::vector<float>& tail_embedding; // Must remain reference (input and output)
    const std::vector<unsigned int> positive_head;
    const std::vector<unsigned int> positive_tail;
    const std::vector<unsigned int> positive_ptr;
    unsigned int n_epochs;
    unsigned int n_head_vertices;
    unsigned int n_tail_vertices;
    const std::vector<float> epochs_per_sample;
    float initial_alpha;
    OptimizerArgs opt_args;
    float negative_sample_rate;
    bool batch;
    std::size_t n_threads;
    std::size_t grain_size;
    bool verbose;

    UmapFactory(bool move_other, bool pcg_rand,
                std::vector<float>& head_embedding,
                std::vector<float>& tail_embedding,
                const std::vector<unsigned int>& positive_head,
                const std::vector<unsigned int>& positive_tail,
                const std::vector<unsigned int>& positive_ptr,
                unsigned int n_epochs, unsigned int n_head_vertices,
                unsigned int n_tail_vertices,
                const std::vector<float>& epochs_per_sample, float initial_alpha,
                OptimizerArgs opt_args, float negative_sample_rate, bool batch,
                std::size_t n_threads, std::size_t grain_size, bool verbose)
        : move_other(move_other), pcg_rand(pcg_rand),
          head_embedding(head_embedding), tail_embedding(tail_embedding),
          positive_head(positive_head), positive_tail(positive_tail),
          positive_ptr(positive_ptr), n_epochs(n_epochs),
          n_head_vertices(n_head_vertices), n_tail_vertices(n_tail_vertices),
          epochs_per_sample(epochs_per_sample), initial_alpha(initial_alpha),
          opt_args(opt_args), negative_sample_rate(negative_sample_rate),
          batch(batch), n_threads(n_threads), grain_size(grain_size), verbose(verbose) {}

    template <typename Gradient>
    void create(const Gradient& gradient, std::mt19937_64 engine) {
        if (move_other) {
            create_impl<true>(gradient, pcg_rand, batch, engine);
        }
        else {
            create_impl<false>(gradient, pcg_rand, batch, engine);
        }
    }

    template <bool DoMove, typename Gradient>
    void create_impl(const Gradient& gradient, bool pcg_rand, bool batch, std::mt19937_64 engine) {
        if (batch) {
            create_impl<BatchRngFactory<true>, DoMove>(gradient, pcg_rand, batch, engine);
        }
        else {
            create_impl<BatchRngFactory<false>, DoMove>(gradient, pcg_rand, batch, engine);
        }
    }

    template <typename BatchRngFactory, bool DoMove, typename Gradient>
    void create_impl(const Gradient& gradient, bool pcg_rand, bool batch, std::mt19937_64 engine) {
        if (pcg_rand) {
            create_impl<typename BatchRngFactory::PcgFactoryType, DoMove>(gradient, batch, engine);
        }
        else {
            create_impl<typename BatchRngFactory::TauFactoryType, DoMove>(gradient, batch, engine);
        }
    }

    std::unique_ptr<uwot::Optimizer> create_optimizer(OptimizerArgs opt_args) {
        float alpha = opt_args.alpha;
        switch (opt_args.opt_method) {
            case OPT_METHOD_SGD:
                if (verbose) {
                    stderr_printf("Optimizing with SGD: alpha = %0.3f\n", alpha);
                }
                return std::make_unique<uwot::Sgd>(alpha);
            default:
                float beta1 = opt_args.beta1;
                float beta2 = opt_args.beta2;
                float eps = opt_args.eps;
                if (verbose) {
                    stderr_printf(
                        "Optimizing with Adam:\n\t alpha = %0.3f,  beta1 = %0.3f, beta2 = %0.3f, eps = %e\n",
                        alpha, beta1, beta2, eps);
                }
                return std::make_unique<uwot::Adam>(alpha, beta1, beta2, eps, head_embedding.size());
        }
    }

    template <typename RandFactory, bool DoMove, typename Gradient>
    void create_impl(const Gradient& gradient, bool batch, std::mt19937_64 engine) {
        uwot::Sampler sampler(epochs_per_sample, negative_sample_rate);
        const std::size_t ndim = head_embedding.size() / n_head_vertices;

        if (batch) {
            auto opt = create_optimizer(opt_args);
            uwot::BatchUpdate<DoMove> update(head_embedding, tail_embedding, std::move(opt));
            uwot::NodeWorker<Gradient, decltype(update), RandFactory> worker(
                gradient, update, positive_head, positive_tail, positive_ptr, sampler,
                ndim, n_tail_vertices);
            create_impl(worker, gradient, engine);
        }
        else {
            uwot::InPlaceUpdate<DoMove> update(head_embedding, tail_embedding, initial_alpha);
            uwot::EdgeWorker<Gradient, decltype(update), RandFactory> worker(
                gradient, update, positive_head, positive_tail, sampler, ndim,
                n_tail_vertices, n_threads);
            create_impl(worker, gradient, engine);
        }
    }

    template <typename Worker, typename Gradient>
    void create_impl(Worker& worker, const Gradient& gradient, std::mt19937_64 engine) {
        if (n_threads > 0) {
            RParallel parallel(n_threads, grain_size);
            create_impl(worker, gradient, parallel, engine);
        }
        else {
            RSerial serial;
            create_impl(worker, gradient, serial, engine);
        }
    }

    template <typename Worker, typename Gradient, typename Parallel>
    void create_impl(Worker& worker, const Gradient& gradient, Parallel& parallel, std::mt19937_64 engine) {
        uwot::optimize_layout(worker, n_epochs, parallel, engine);
    }
};

#endif //ACTIONET_UMAPFACTORY_HPP
