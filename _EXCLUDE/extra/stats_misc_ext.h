#ifndef STATS_MISC_EXT_H
#define STATS_MISC_EXT_H

#include "libactionet_config.hpp"
#include <pcg/pcg_random.hpp>

arma::mat sampleUnif(int l, int m, double a, double b, int seed);

void randNorm_inplace(int n, double *out, int seed);

uint32_t lfsr113(uint64_t **state);

void lfsr113_seed(uint32_t seed, uint64_t **state);

void randN_Marsaglia(double *values, int n, pcg32 rng);

void randN_BM(double *values, int n, uint64_t **state);

void randN_normsinv(double *values, int n);

#endif //STATS_MISC_EXT_H
