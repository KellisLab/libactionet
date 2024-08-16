#ifndef ACTIONET_UTILS_XMAP_HPP
#define ACTIONET_UTILS_XMAP_HPP

#include "libactionet_config.hpp"
#include "utils_internal/umap_structs.hpp"

// Functions
void create_umap(UmapFactory &umap_factory, double a, double b, double gamma, bool approx_pow);

void create_tumap(UmapFactory &umap_factory);

void create_pacmap(UmapFactory &umap_factory, double a, double b);

void create_largevis(UmapFactory &umap_factory, double gamma);


#endif //ACTIONET_UTILS_XMAP_HPP
