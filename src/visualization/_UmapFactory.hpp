// Modified variant of required UmapFactory struct implemented in uwot Rcpp interface.
// Calls and controls uwot
// Key modifications:
//      Eliminated R-dependency.
//      Remove support for R callback.
//      Added shared RNG engine for reproducibility in multithreaded operation.

#ifndef ACTIONET_UMAPFACTORY_HPP
#define ACTIONET_UMAPFACTORY_HPP

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

#include "libactionet_config.hpp"
#include "uwot/epoch.h"
#include "uwot/optimize.h"
#include "uwot/sampler.h"

// MIT-licensed replacements for the historical uwot/rng.h (GPLv3-or-later) and
// uwot/rparallel.h (GPLv3-or-later, transitively RcppPerpendicular GPLv2+).
// Provide the same public symbols expected below:
//   pcg_prng, batch_pcg_factory, pcg_factory,
//   batch_tau_factory, tau_factory, deterministic_factory,
//   RParallel, RSerial.
#include "_uwot_rng.hpp"
#include "_uwot_parallel.hpp"
#include "visualization/OptimizerArgs.hpp"

// Template class specialization to handle different rng/batch combinations
template <bool DoBatch = true>
struct BatchRngFactory {
    using PcgFactoryType = batch_pcg_factory;
    using TauFactoryType = batch_tau_factory;
    using DeterministicFactoryType = deterministic_factory;
};

template <>
struct BatchRngFactory<false> {
    using PcgFactoryType = pcg_factory;
    using TauFactoryType = tau_factory;
    using DeterministicFactoryType = deterministic_factory;
};

struct NoOpProgress {
    explicit NoOpProgress(bool) {}

    bool is_aborted() const { return false; }

    void report() const {}
};

struct UmapFactory {
    bool move_other;
    std::string rng_type;
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

    UmapFactory(bool move_other, std::string rng_type,
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
        : move_other(move_other), rng_type(normalize_rng_type(std::move(rng_type))),
          head_embedding(head_embedding), tail_embedding(tail_embedding),
          positive_head(positive_head), positive_tail(positive_tail),
          positive_ptr(positive_ptr), n_epochs(n_epochs),
          n_head_vertices(n_head_vertices), n_tail_vertices(n_tail_vertices),
          epochs_per_sample(epochs_per_sample), initial_alpha(initial_alpha),
          opt_args(opt_args), negative_sample_rate(negative_sample_rate),
          batch(batch), n_threads(n_threads), grain_size(grain_size), verbose(verbose) {}

    template <typename Gradient>
    void create(const Gradient& gradient, std::mt19937_64& engine) {
        if (move_other) {
            create_impl<true>(gradient, rng_type, batch, engine);
        }
        else {
            create_impl<false>(gradient, rng_type, batch, engine);
        }
    }

    template <bool DoMove, typename Gradient>
    void create_impl(const Gradient& gradient, const std::string& rng_type,
                     bool batch, std::mt19937_64& engine) {
        if (batch) {
            create_impl<BatchRngFactory<true>, DoMove>(gradient, rng_type, batch, engine);
        }
        else {
            create_impl<BatchRngFactory<false>, DoMove>(gradient, rng_type, batch, engine);
        }
    }

    template <typename BatchRngFactory, bool DoMove, typename Gradient>
    void create_impl(const Gradient& gradient, const std::string& rng_type,
                     bool batch, std::mt19937_64& engine) {
        if (rng_type == "pcg") {
            create_impl<typename BatchRngFactory::PcgFactoryType, DoMove>(gradient, batch, engine);
        }
        else if (rng_type == "tausworthe") {
            create_impl<typename BatchRngFactory::TauFactoryType, DoMove>(gradient, batch, engine);
        }
        else if (rng_type == "deterministic") {
            create_impl<typename BatchRngFactory::DeterministicFactoryType, DoMove>(gradient, batch, engine);
        }
        else {
            throw std::invalid_argument("Invalid rng_type. Must be one of: pcg, tausworthe, deterministic");
        }
    }

    std::unique_ptr<uwot::Optimizer> create_optimizer() {
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
                        "Optimizing with Adam:\n\t alpha = %0.3f,  beta1 = %0.3f, beta2 = %0.3f, eps = %0.3e\n",
                        alpha, beta1, beta2, eps);
                }
                return std::make_unique<uwot::Adam>(alpha, beta1, beta2, eps, head_embedding.size());
        }
    }

    template <typename RandFactory, bool DoMove, typename Gradient>
    void create_impl(const Gradient& gradient, bool batch, std::mt19937_64& engine) {
        uwot::Sampler sampler(epochs_per_sample, negative_sample_rate);
        const std::size_t ndim = head_embedding.size() / n_head_vertices;

        auto epoch_callback = std::make_unique<uwot::DoNothingCallback>();
        if (batch) {
            auto opt = create_optimizer();
            uwot::BatchUpdate<DoMove> update(head_embedding, tail_embedding, std::move(opt), epoch_callback.release());
            uwot::NodeWorker<Gradient, decltype(update), RandFactory> worker(
                gradient, update, positive_head, positive_tail, positive_ptr, sampler,
                ndim, n_tail_vertices);
            create_impl(worker, gradient, engine);
        }
        else {
            uwot::InPlaceUpdate<DoMove> update(head_embedding, tail_embedding, initial_alpha, epoch_callback.release());
            uwot::EdgeWorker<Gradient, decltype(update), RandFactory> worker(
                gradient, update, positive_head, positive_tail, sampler, ndim,
                n_tail_vertices, n_threads);
            create_impl(worker, gradient, engine);
        }
    }

    template <typename Worker, typename Gradient>
    void create_impl(Worker& worker, const Gradient& gradient, std::mt19937_64& engine) {
        NoOpProgress progress(verbose);
        if (n_threads > 0) {
            RParallel parallel(n_threads, grain_size);
            create_impl(worker, gradient, progress, parallel, engine);
        }
        else {
            RSerial serial;
            create_impl(worker, gradient, progress, serial, engine);
        }
    }

    template <typename Worker, typename Gradient, typename Progress, typename Parallel>
    void create_impl(Worker& worker, const Gradient&, Progress& progress, Parallel& parallel, std::mt19937_64& engine) {
        uwot::optimize_layout(worker, progress, n_epochs, parallel, engine);
    }

private:
    static std::string normalize_rng_type(std::string rng_type) {
        std::transform(rng_type.begin(), rng_type.end(), rng_type.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return rng_type;
    }
};

#endif //ACTIONET_UMAPFACTORY_HPP
