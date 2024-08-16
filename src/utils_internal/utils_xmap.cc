//#include "utils_internal/utils_xmap.hpp"
//
//void create_umap(UmapFactory &umap_factory, double a, double b, double gamma, bool approx_pow) {
//    if (approx_pow) {
//        const uwot::apumap_gradient gradient(a, b, gamma);
//        umap_factory.create(gradient);
//    } else {
//        const uwot::umap_gradient gradient(a, b, gamma);
//        umap_factory.create(gradient);
//    }
//}
//
//void create_tumap(UmapFactory &umap_factory) {
//    const uwot::tumap_gradient gradient;
//    umap_factory.create(gradient);
//}
//
//void create_pacmap(UmapFactory &umap_factory, double a, double b) {
//    const uwot::pacmap_gradient gradient(a, b);
//    umap_factory.create(gradient);
//}
//
//void create_largevis(UmapFactory &umap_factory, double gamma) {
//    const uwot::largevis_gradient gradient(gamma);
//    umap_factory.create(gradient);
//}
//
