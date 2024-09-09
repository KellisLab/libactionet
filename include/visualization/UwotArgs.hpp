#ifndef ACTIONET_UWOTARGS_HPP
#define ACTIONET_UWOTARGS_HPP

#include "libactionet_config.hpp"
#include "find_ab.hpp"
#include "OptimizerArgs.hpp"

// Constants
constexpr int METHOD_UMAP = 1;
constexpr int METHOD_TUMAP = 2;
constexpr int METHOD_LARGEVIZ = 3;

// User-specified arguments
class UwotArgs {
public:
    // Defaults
    int n_components = 2;
    float spread = 1;
    float min_dist = 1;
    int n_epochs = 0;
    float learning_rate = OPT_ALPHA; // Passed to optimizer as `alpha`
    float repulsion_strength = 1; // alias for `gamma`
    float negative_sample_rate = 5; // alias for `alpha`
    std::string method = "umap";
    bool approx_pow = false;
    bool pcg_rand = true;
    bool batch = true;
    // int seed = 0;
    std::size_t n_threads = 1;
    std::size_t grain_size = 1;
    bool verbose = true;
    OptimizerArgs opt_args = OptimizerArgs(learning_rate);
    // Initialized by members
    float a = 0; // Dummy value. Overwritten by initializer.
    float b = 0; // Dummy value. Overwritten by initializer.
    float& alpha = learning_rate; // alias
    // CAUTON: NOT necessarily the same `alpha` as `opt_args`. `opt_args` may be initialized separately.
    float& gamma = repulsion_strength; // alias
    // std::mt19937_64 engine;
private:
    int cost_func = 0; // Dummy value. Overwritten by initializer.

public:
    // Default constructor
    UwotArgs() {
        set_method(method);
        // set_seed(seed);
    }

    // Minimal constructor
    explicit UwotArgs(std::string method) {
        set_method(method);
    }

    // Full constructor
    // UwotArgs(int n_components, float spread, float min_dist, int n_epochs, float learning_rate, float repulsion_strength,
    //          float negative_sample_rate, std::size_t grain_size, bool approx_pow, bool pcg_rand, bool batch, int seed,
    //          OptimizerArgs opt_args)
    //     : n_components(n_components), spread(spread), min_dist(min_dist), n_epochs(n_epochs), learning_rate(learning_rate),
    //       repulsion_strength(repulsion_strength), negative_sample_rate(negative_sample_rate), grain_size(grain_size),
    //       approx_pow(approx_pow), pcg_rand(pcg_rand), batch(batch), seed(seed), opt_args(opt_args) {
    //     set_ab();
    //     set_seed(seed);
    // }
    UwotArgs(
        int n_components,
        float spread,
        float min_dist,
        int n_epochs,
        float learning_rate,
        float repulsion_strength,
        float negative_sample_rate,
        std::string& method,
        bool approx_pow,
        bool pcg_rand,
        bool batch,
        std::size_t n_threads,
        std::size_t grain_size,
        bool verbose,
        OptimizerArgs opt_args
    )
        : n_components(n_components),
          spread(spread),
          min_dist(min_dist),
          n_epochs(n_epochs),
          learning_rate(learning_rate),
          repulsion_strength(repulsion_strength),
          negative_sample_rate(negative_sample_rate),
          method(method),
          approx_pow(approx_pow),
          pcg_rand(pcg_rand),
          batch(batch),
          n_threads(n_threads),
          grain_size(grain_size),
          verbose(verbose),
          opt_args(opt_args) {
        // set_ab();
        set_method(method);
        // set_seed(seed);
    }

    // void set_seed(const int seed) {
    //     engine = std::mt19937_64(seed);
    // }

    void set_method(const std::string& method) {
        if (method == "tumap") {
            cost_func = METHOD_TUMAP;
            set_ab(1, 1);
        }
        else if (method == "largevis") {
            cost_func = METHOD_LARGEVIZ;
        }
        else {
            cost_func = METHOD_UMAP;
            set_ab();
        }
    }

    int get_cost_func() const {
        return cost_func;
    }

    void set_ab() {
        auto [fst, snd] = find_ab(spread, min_dist);
        a = fst;
        b = snd;
    }

    void set_ab(const float a, const float b) {
        this->a = a;
        this->b = b;
    }

    void set_OptimizerArgs(const OptimizerArgs& opt_args) {
        this->opt_args = opt_args;
    }
};

#endif //ACTIONET_UWOTARGS_HPP
