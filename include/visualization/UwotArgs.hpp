// Class defining parameters controlling uwot interface
#ifndef ACTIONET_UWOTARGS_HPP
#define ACTIONET_UWOTARGS_HPP

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>
#include <vector>

#include "libactionet_config.hpp"
#include "find_ab.hpp"
#include "OptimizerArgs.hpp"

// Constants
constexpr int METHOD_UMAP = 1;
constexpr int METHOD_TUMAP = 2;
constexpr int METHOD_LARGEVIZ = 3;
constexpr int METHOD_LEOPOLD = 4;
constexpr int METHOD_LEOPOLD2 = 5;

// User-specified arguments
class UwotArgs {
public:
    // Defaults
    unsigned int n_components = 2;
    float spread = 1;
    float min_dist = 1;
    unsigned int n_epochs = 0;
    float learning_rate = LR_OPT_ALPHA; // Passed to optimizer as `alpha` if `opt_args.alpha` unspecified.
    float repulsion_strength = 1; // alias for `gamma`
    float negative_sample_rate = 5; // alias for `alpha`
    bool approx_pow = false;
    bool pcg_rand = true;
    bool batch = true;
    std::size_t n_threads = 1;
    std::size_t grain_size = 1;
    bool verbose = true;
    // C++-only diagnostic switch for verbose runtime environment details.
    // This is intentionally not exposed via R/Python wrapper APIs.
    bool debug_runtime_diagnostics = false;
    OptimizerArgs opt_args = OptimizerArgs(learning_rate);
    // Initialized by members
    float a = 0; // Dummy value. Overwritten by initializer.
    float b = 0; // Dummy value. Overwritten by initializer.
    // CAUTION: NOT necessarily the same `alpha` as `opt_args`. `opt_args` may be initialized separately.
    // Changing `opt_args.alpha` should not change this value.
    float& alpha = learning_rate; // alias. Passed to UmapFactory()
    float& gamma = repulsion_strength; // alias. Passed to UmapFactory.create()
    // Required when method is "leopold"/"leopold2"
    std::vector<float> ai;
    std::vector<float> aj;
private:
    int cost_func = 0; // Dummy value. Overwritten by initializer.
    std::string method = "umap";
    std::string rng_type = "pcg"; // "pcg", "tausworthe", "deterministic"
    int seed = 0;
    std::mt19937_64 engine;
    bool rng_type_explicit = false;

public:
    // Default constructor
    UwotArgs() {
        set_method(method);
        set_ab();
        set_seed(seed);
        sync_rng_type_from_legacy_flag();
    }

    // Minimal constructor
    explicit UwotArgs(const std::string& method) {
        set_method(method);
        set_ab();
        set_seed(seed);
        sync_rng_type_from_legacy_flag();
    }

    // Full constructor
    UwotArgs(
        const std::string& method,
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
        OptimizerArgs opt_args,
        const std::string& rng_type = ""
    )
        : n_components(n_components),
          spread(spread),
          min_dist(min_dist),
          n_epochs(n_epochs),
          learning_rate(learning_rate),
          repulsion_strength(repulsion_strength),
          negative_sample_rate(negative_sample_rate),
          approx_pow(approx_pow),
          pcg_rand(pcg_rand),
          batch(batch),
          n_threads(n_threads),
          grain_size(grain_size),
          verbose(verbose),
          opt_args(opt_args),
          method(method),
          seed(seed) {
        set_method(method);
        set_ab();
        set_seed(seed);
        if (rng_type.empty()) {
            sync_rng_type_from_legacy_flag();
        }
        else {
            set_rng_type(rng_type);
        }
    }

    void set_seed(const int seed) {
        this->seed = seed;
        this->engine = std::mt19937_64(seed);
    }

    int get_seed() const {
        return seed;
    }

    std::mt19937_64& get_engine() {
        return engine;
    }

    const std::mt19937_64& get_engine() const {
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
        const auto method_norm = normalize_lower(method);
        this->method = method_norm;
        if (method_norm == "umap") {
            this->cost_func = METHOD_UMAP;
        }
        else if (method_norm == "tumap") {
            this->cost_func = METHOD_TUMAP;
            set_ab(1, 1); // Automatically by uwot, but just in case.
        }
        else if (method_norm == "largevis") {
            this->cost_func = METHOD_LARGEVIZ;
        }
        else if (method_norm == "leopold") {
            this->cost_func = METHOD_LEOPOLD;
        }
        else if (method_norm == "leopold2") {
            this->cost_func = METHOD_LEOPOLD2;
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

    void set_rng_type(const std::string& rng_type) {
        const auto rng_norm = normalize_lower(rng_type);
        if (rng_norm != "pcg" && rng_norm != "tausworthe" && rng_norm != "deterministic") {
            throw std::invalid_argument("Invalid 'rng_type'. Must be one of: pcg, tausworthe, deterministic");
        }
        this->rng_type = rng_norm;
        this->rng_type_explicit = true;
    }

    std::string get_rng_type() const {
        if (rng_type_explicit) {
            return rng_type;
        }
        return pcg_rand ? "pcg" : "tausworthe";
    }

private:
    static std::string normalize_lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    void sync_rng_type_from_legacy_flag() {
        this->rng_type = pcg_rand ? "pcg" : "tausworthe";
        this->rng_type_explicit = false;
    }
};

#endif //ACTIONET_UWOTARGS_HPP
