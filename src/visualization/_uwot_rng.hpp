// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026 MIT CompBio Group
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// -----------------------------------------------------------------------------
// Clean-room MIT reimplementation of the RNG factory surface required by
// uwot's optimize_layout() driver (see _UmapFactory.hpp and uwot/epoch.h).
//
// Replaces uwot/rng.h (GPLv3-or-later) and its dqrng::convert_seed dependency.
//
// Public types (matches the historical rng.h contract expected by
// _UmapFactory.hpp and uwot::{EdgeWorker,NodeWorker}):
//   - pcg_prng                — thin adaptor around pcg32 exposing operator()(n, _, _)
//   - batch_pcg_factory       — per-worker pcg32 factory with parallel-safe seeds
//   - pcg_factory             — single-shared-seed pcg32 factory (non-batch mode)
//   - batch_tau_factory       — per-worker uwot::tau_prng factory
//   - tau_factory             — single-shared-seed uwot::tau_prng factory
//   - deterministic_factory   — thin factory around uwot::deterministic_ng
//
// Dependencies:
//   - pcg/pcg_random.hpp      (Apache-2.0)
//   - uwot/tauprng.h          (BSD-2-Clause)
// No dqrng, no Rcpp*, no GPL headers reachable from this file.

#ifndef ACTIONET_UWOT_RNG_HPP
#define ACTIONET_UWOT_RNG_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "pcg/pcg_random.hpp"
#include "uwot/tauprng.h"

namespace actionet {
namespace uwot_rng {

// Compose two uint32 seeds into a single uint64 by simple bit-packing. This is
// a trivial arithmetic operation, not derivative of any specific implementation.
inline std::uint64_t pack_seeds(std::uint32_t hi, std::uint32_t lo) noexcept {
    return (static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo);
}

// Draw a uniformly distributed value from a std::mt19937_64. Using
// std::uniform_int_distribution here (rather than StatsLib) keeps the seed
// factories self-contained and free of Armadillo/StatsLib pulls.
inline std::uint64_t draw_u64(std::mt19937_64& engine) {
    std::uniform_int_distribution<std::uint64_t> dist(
        0, (std::numeric_limits<std::uint64_t>::max)());
    return dist(engine);
}

inline std::uint32_t draw_u32(std::mt19937_64& engine) {
    std::uniform_int_distribution<std::uint32_t> dist(
        0, (std::numeric_limits<std::uint32_t>::max)());
    return dist(engine);
}

} // namespace uwot_rng
} // namespace actionet


// The following names intentionally live in the global namespace, matching the
// historical contract exported by uwot/rng.h and consumed by _UmapFactory.hpp.

struct pcg_prng {
    pcg32 gen;

    explicit pcg_prng(std::uint64_t seed) { gen.seed(seed); }

    // Contract inherited from uwot: return a value in [0, n).
    // The (edge_index, epoch) parameters are ignored for stochastic PRNGs;
    // deterministic_ng uses them (see uwot/tauprng.h).
    inline std::size_t operator()(std::size_t n, std::size_t /*edge_index*/,
                                  std::size_t /*epoch*/) {
        return static_cast<std::size_t>(gen(static_cast<std::uint32_t>(n)));
    }
};

struct batch_pcg_factory {
    static constexpr std::size_t seeds_per_rng = 2;

    std::size_t              n_rngs;
    std::vector<std::uint32_t> seeds;

    batch_pcg_factory() : n_rngs(1), seeds(seeds_per_rng * n_rngs) {}
    explicit batch_pcg_factory(std::size_t n_rngs_)
        : n_rngs(n_rngs_), seeds(seeds_per_rng * n_rngs_) {}

    void reseed(std::mt19937_64& engine) {
        for (auto& s : seeds) {
            s = actionet::uwot_rng::draw_u32(engine);
        }
    }

    pcg_prng create(std::size_t n) {
        const std::size_t idx = n * seeds_per_rng;
        return pcg_prng(actionet::uwot_rng::pack_seeds(seeds[idx], seeds[idx + 1]));
    }
};

struct pcg_factory {
    std::uint32_t seed1;

    explicit pcg_factory(std::size_t /*n_rngs*/) : seed1(0) {}

    void reseed(std::mt19937_64& engine) {
        seed1 = actionet::uwot_rng::draw_u32(engine);
    }

    pcg_prng create(std::size_t seed) {
        return pcg_prng(
            actionet::uwot_rng::pack_seeds(seed1, static_cast<std::uint32_t>(seed)));
    }
};

struct batch_tau_factory {
    static constexpr std::size_t seeds_per_rng = 3;

    std::size_t              n_rngs;
    std::vector<std::uint64_t> seeds;

    batch_tau_factory() : n_rngs(1), seeds(seeds_per_rng * n_rngs) {}
    explicit batch_tau_factory(std::size_t n_rngs_)
        : n_rngs(n_rngs_), seeds(seeds_per_rng * n_rngs_) {}

    void reseed(std::mt19937_64& engine) {
        for (auto& s : seeds) {
            s = actionet::uwot_rng::draw_u64(engine);
        }
    }

    uwot::tau_prng create(std::size_t n) {
        const std::size_t idx = n * seeds_per_rng;
        return uwot::tau_prng(seeds[idx], seeds[idx + 1], seeds[idx + 2]);
    }
};

struct tau_factory {
    std::uint64_t seed1;
    std::uint64_t seed2;

    explicit tau_factory(std::size_t /*n_rngs*/) : seed1(0), seed2(0) {}

    void reseed(std::mt19937_64& engine) {
        seed1 = actionet::uwot_rng::draw_u64(engine);
        seed2 = actionet::uwot_rng::draw_u64(engine);
    }

    uwot::tau_prng create(std::size_t seed) {
        return uwot::tau_prng(seed1, seed2, static_cast<std::uint64_t>(seed));
    }
};

struct deterministic_factory {
    explicit deterministic_factory(std::size_t /*n_rngs*/) {}

    void reseed(std::mt19937_64& /*engine*/) {}

    uwot::deterministic_ng create(std::size_t /*n*/) {
        return uwot::deterministic_ng();
    }
};

#endif // ACTIONET_UWOT_RNG_HPP
