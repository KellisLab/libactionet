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
    unsigned int n_components = 2;
    float spread = 1;
    float min_dist = 1;
    unsigned int n_epochs = 0;
    float learning_rate = OPT_ALPHA; // Passed to optimizer as `alpha`
    float repulsion_strength = 1; // alias for `gamma`
    float negative_sample_rate = 5; // alias for `alpha`
    bool approx_pow = false;
    bool pcg_rand = true;
    bool batch = true;
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
private:
    int cost_func = 0; // Dummy value. Overwritten by initializer.
    std::string method = "umap";
    int seed = 0;
    std::mt19937_64 engine;

public:
    // Default constructor
    UwotArgs() {
        set_method(method);
        set_ab();
        set_seed(seed);
    }

    // Minimal constructor
    explicit UwotArgs(const std::string& method) {
        set_method(method);
        set_seed(seed);
    }

    // Full constructor
    UwotArgs(
        std::string& method,
        unsigned int n_components,
        float spread,
        float min_dist,
        unsigned int n_epochs,
        float learning_rate,
        float repulsion_strength,
        float negative_sample_rate,
        bool approx_pow,
        bool pcg_rand,
        bool batch,
        int seed,
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
          seed(seed),
          n_threads(n_threads),
          grain_size(grain_size),
          verbose(verbose),
          opt_args(opt_args) {
        set_method(method);
        set_ab();
        set_seed(seed);
    }

    void set_seed(const int seed) {
        this->engine = std::mt19937_64(seed);
    }

    int get_seed() const {
        return seed;
    }

    std::mt19937_64 get_engine() const {
        return engine;
    }

    int get_cost_func() const {
        return cost_func;
    }

    void set_ab() {
        auto [fst, snd] = find_ab(spread, min_dist);
        this->a = fst;
        this->b = snd;
    }

    void set_ab(const float a, const float b) {
        this->a = a;
        this->b = b;
    }

    // void set_OptimizerArgs(const OptimizerArgs& opt_args) {
    //     this->opt_args = opt_args;
    // }
    //
    // OptimizerArgs get_OptimizerArgs() const {
    //     return opt_args;
    // }

    void set_method(const std::string& method) {
        this->method = method;
        if (method == "umap") {
            this->cost_func = METHOD_UMAP;
        }
        else if (method == "tumap") {
            this->cost_func = METHOD_TUMAP;
            set_ab(1,1); // Unused and set automatically by uwot, but just in case.
        }
        else if (method == "largevis") {
            this->cost_func = METHOD_LARGEVIZ;
        }
        else {
            stderr_printf("Invalid 'method'. Defaulting to 'umap'\n");
            this->method = "umap";
            this->cost_func = METHOD_UMAP;
        }
    }

    std::string get_method() const {
        return method;
    }
};

#endif //ACTIONET_UWOTARGS_HPP
