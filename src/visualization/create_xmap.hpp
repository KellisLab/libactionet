#ifndef CREATE_XMAP_HPP
#define CREATE_XMAP_HPP

#include "UmapFactory.hpp"
#include "uwot/gradient.h"

// Functions
void create_umap(UmapFactory &umap_factory, double a, double b, double gamma, bool approx_pow);

void create_tumap(UmapFactory &umap_factory);

void create_pacmap(UmapFactory &umap_factory, double a, double b);

void create_largevis(UmapFactory &umap_factory, double gamma);

#endif
