//  UWOT -- An R package for dimensionality reduction using UMAP
//
//  Copyright (C) 2018 James Melville
//
//  This file is part of UWOT
//
//  UWOT is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  UWOT is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with UWOT.  If not, see <http://www.gnu.org/licenses/>.

#ifndef UWOT_RNG_H
#define UWOT_RNG_H

#include <limits>

// linked from dqrng
#include "convert_seed.h"
#include "pcg/pcg_random.hpp"

#include "uwot/tauprng.h"
#include <StatsLib/stats.hpp>

static uint64_t random64(std::mt19937_64& engine) {
    return static_cast<uint64_t>(
        stats::runif(0, (std::numeric_limits<uint64_t>::max)(), engine));
}

static uint32_t random32(std::mt19937_64& engine) {
    return static_cast<uint32_t>(
        stats::runif(0, (std::numeric_limits<uint32_t>::max)(), engine));
}

struct batch_tau_factory {
    std::size_t n_rngs;
    std::vector<uint64_t> seeds;
    static const constexpr std::size_t seeds_per_rng = 3;

    batch_tau_factory() : n_rngs(1), seeds(seeds_per_rng * n_rngs) {}

    batch_tau_factory(std::size_t n_rngs)
        : n_rngs(n_rngs), seeds(seeds_per_rng * n_rngs) {}

    void reseed() {
        std::mt19937_64 engine;
        for (std::size_t i = 0; i < seeds.size(); i++) {
            seeds[i] = random64(engine);
        }
    }

    uwot::tau_prng create(std::size_t n) {
        const std::size_t idx = n * seeds_per_rng;
        return uwot::tau_prng(seeds[idx], seeds[idx + 1], seeds[idx + 2]);
    }
};

struct pcg_prng {
    pcg32 gen;

    pcg_prng(uint64_t seed) { gen.seed(seed); }

    // return a value in (0, n]
    inline std::size_t operator()(std::size_t n) {
        std::size_t result = gen(n);
        return result;
    }
};

struct batch_pcg_factory {
    std::size_t n_rngs;
    std::vector<uint32_t> seeds;
    static const constexpr std::size_t seeds_per_rng = 2;

    batch_pcg_factory() : n_rngs(1), seeds(seeds_per_rng * n_rngs) {}

    batch_pcg_factory(std::size_t n_rngs)
        : n_rngs(n_rngs), seeds(seeds_per_rng * n_rngs) {}

    void reseed() {
        std::mt19937_64 engine;
        for (std::size_t i = 0; i < seeds.size(); i++) {
            seeds[i] = random32(engine);
        }
    }

    pcg_prng create(std::size_t n) {
        uint32_t pcg_seeds[2] = {
            seeds[n * seeds_per_rng],
            seeds[n * seeds_per_rng + 1]
        };
        return pcg_prng(dqrng::convert_seed<uint64_t>(pcg_seeds, 2));
    }
};

// For backwards compatibility in non-batch mode
struct tau_factory {
    uint64_t seed1;
    uint64_t seed2;
    tau_factory(std::size_t) : seed1(0), seed2(0) {}

    void reseed() {
        std::mt19937_64 engine;
        seed1 = random64(engine);
        seed2 = random64(engine);
    }

    uwot::tau_prng create(std::size_t seed) {
        return uwot::tau_prng(seed1, seed2, uint64_t{seed});
    }
};

struct pcg_factory {
    uint32_t seed1;
    pcg_factory(std::size_t) : seed1(0) {}

    void reseed() {
        std::mt19937_64 engine;
        seed1 = random32(engine);
    }

    pcg_prng create(std::size_t seed) {
        uint32_t seeds[2] = {seed1, static_cast<uint32_t>(seed)};
        return pcg_prng(dqrng::convert_seed<uint64_t>(seeds, 2));
    }
};

#endif // UWOT_RNG_H
