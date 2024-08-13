#ifndef LIBACTIONET_UMAP_STRUCTS_HPP
#define LIBACTIONET_UMAP_STRUCTS_HPP

#include "libactionet_config.hpp"
#include "uwot/optimize.h"
#include "uwot/rng.h"
#include "uwot/epoch.h"
#include "uwot/rparallel.h"

// Template class specialization to handle different rng/batch combinations
template<bool DoBatch = true>
struct BatchRngFactory {
    using PcgFactoryType = batch_pcg_factory;
    using TauFactoryType = batch_tau_factory;
};
template<>
struct BatchRngFactory<false> {
    using PcgFactoryType = pcg_factory;
    using TauFactoryType = tau_factory;
};

struct UmapFactory {
    bool move_other;
    bool pcg_rand;
    std::vector<float> &head_embedding;
    std::vector<float> &tail_embedding;
    const std::vector<unsigned int> &positive_head;
    const std::vector<unsigned int> &positive_tail;
    const std::vector<unsigned int> &positive_ptr;
    unsigned int n_epochs;
    unsigned int n_head_vertices;
    unsigned int n_tail_vertices;
    const std::vector<float> &epochs_per_sample;
    float initial_alpha;
    float negative_sample_rate;
    bool batch;
    std::size_t n_threads;
    std::size_t grain_size;
    std::string opt_name;
    double alpha, beta1, beta2, eps;
    std::mt19937_64 engine;

    UmapFactory(bool move_other, bool pcg_rand,
                std::vector<float> &head_embedding,
                std::vector<float> &tail_embedding,
                const std::vector<unsigned int> &positive_head,
                const std::vector<unsigned int> &positive_tail,
                const std::vector<unsigned int> &positive_ptr,
                unsigned int n_epochs, unsigned int n_head_vertices,
                unsigned int n_tail_vertices,
                const std::vector<float> &epochs_per_sample, float initial_alpha,
                float negative_sample_rate, bool batch, std::size_t n_threads,
                std::size_t grain_size, std::string opt_name, double alpha,
                double beta1, double beta2, double eps, std::mt19937_64 &engine)
            : move_other(move_other),
              pcg_rand(pcg_rand),
              head_embedding(head_embedding),
              tail_embedding(tail_embedding),
              positive_head(positive_head),
              positive_tail(positive_tail),
              positive_ptr(positive_ptr),
              n_epochs(n_epochs),
              n_head_vertices(n_head_vertices),
              n_tail_vertices(n_tail_vertices),
              epochs_per_sample(epochs_per_sample),
              initial_alpha(initial_alpha),
              negative_sample_rate(negative_sample_rate),
              batch(batch),
              n_threads(n_threads),
              grain_size(grain_size),
              alpha(alpha),
              beta1(beta1),
              beta2(beta2),
              eps(eps),
              engine(engine),
              opt_name(opt_name) {}

    template<typename Gradient>
    void create(const Gradient &gradient) {
        if (move_other) {
            create_impl<true>(gradient, pcg_rand, batch);
        } else {
            create_impl<false>(gradient, pcg_rand, batch);
        }
    }

    template<bool DoMove, typename Gradient>
    void create_impl(const Gradient &gradient, bool pcg_rand, bool batch) {
        if (batch) {
            create_impl<BatchRngFactory<true>, DoMove>(gradient, pcg_rand, batch);
        } else {
            create_impl<BatchRngFactory<false>, DoMove>(gradient, pcg_rand, batch);
        }
    }

    template<typename BatchRngFactory, bool DoMove, typename Gradient>
    void create_impl(const Gradient &gradient, bool pcg_rand, bool batch) {
        if (pcg_rand) {
            create_impl<typename BatchRngFactory::PcgFactoryType, DoMove>(gradient,
                                                                          batch);
        } else {
            create_impl<typename BatchRngFactory::TauFactoryType, DoMove>(gradient,
                                                                          batch);
        }
    }

    uwot::Adam create_adam() {
        return uwot::Adam(alpha, beta1, beta2, eps, head_embedding.size());
    }

    uwot::Sgd create_sgd() { return uwot::Sgd(alpha); }

    template<typename RandFactory, bool DoMove, typename Gradient>
    void create_impl(const Gradient &gradient, bool batch) {
        if (batch) {
            if (opt_name == "adam") {
                auto opt = create_adam();
                create_impl_batch_opt<decltype(opt), RandFactory, DoMove, Gradient>(
                        gradient, opt, batch);
            } else if (opt_name == "sgd") {
                auto opt = create_sgd();
                create_impl_batch_opt<decltype(opt), RandFactory, DoMove, Gradient>(
                        gradient, opt, batch);
            } else {
                stderr_printf("Unknown optimization method: %s\n", opt_name.c_str());
                FLUSH;
                return;
            }
        } else {
            const std::size_t ndim = head_embedding.size() / n_head_vertices;
            uwot::Sampler sampler(epochs_per_sample, negative_sample_rate);
            uwot::InPlaceUpdate<DoMove> update(head_embedding, tail_embedding,
                                               initial_alpha);
            uwot::EdgeWorker<Gradient, decltype(update), RandFactory> worker(
                    gradient, update, positive_head, positive_tail, sampler, ndim,
                    n_tail_vertices, n_threads, engine);
            create_impl(worker, gradient);
        }
    }

    template<typename Opt, typename RandFactory, bool DoMove, typename Gradient>
    void create_impl_batch_opt(const Gradient &gradient, Opt &opt, bool batch) {
        uwot::Sampler sampler(epochs_per_sample, negative_sample_rate);
        const std::size_t ndim = head_embedding.size() / n_head_vertices;
        uwot::BatchUpdate<DoMove, decltype(opt)> update(head_embedding,
                                                        tail_embedding, opt);
        uwot::NodeWorker<Gradient, decltype(update), RandFactory> worker(
                gradient, update, positive_head, positive_tail, positive_ptr, sampler,
                ndim, n_tail_vertices, engine);
        create_impl(worker, gradient);
    }

    template<typename Worker, typename Gradient>
    void create_impl(Worker &worker, const Gradient &gradient) {
        if (n_threads > 0) {
            RParallel parallel(n_threads, grain_size);
            create_impl(worker, gradient, parallel);
        } else {
            RSerial serial;
            create_impl(worker, gradient, serial);
        }
    }

    template<typename Worker, typename Gradient, typename Parallel>
    void create_impl(Worker &worker, const Gradient &gradient,
                     Parallel &parallel) {
        uwot::optimize_layout(worker, n_epochs, parallel);
    }
};

#endif //LIBACTIONET_UMAP_STRUCTS_HPP
