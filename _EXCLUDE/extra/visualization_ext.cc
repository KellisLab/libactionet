#include "visualization_ext.h"

const double UMAP_A[101] = {
        1.93280839781719, 1.89560586588002, 1.85873666431227, 1.82221007490834,
        1.78603612060048, 1.75022496320214, 1.71478579945151, 1.67972997626197,
        1.64506544270902, 1.610800661285, 1.57694346052399, 1.54350101780511,
        1.51047986323257, 1.47788588612333, 1.44572435168023, 1.41399925414561,
        1.38271638006498, 1.35187804260518, 1.3214872860387, 1.29154663185922,
        1.26205810311418, 1.23302325071067, 1.20444317424075, 1.17631854866857,
        1.14864964274379, 1.12143634262879, 1.09467817152021, 1.0683743100033,
        1.04252361298475, 1.01712481754341, 0.992175611624647, 0.967674513244996,
        0.943619207179927, 0.920007077834315, 0.896835219021839, 0.874100443595699,
        0.851800999392949, 0.829931994792615, 0.808490430178554, 0.787472613514984,
        0.766873638278737, 0.746690990400437, 0.726919886947928, 0.707556026044195,
        0.688594985599233, 0.670032232635194, 0.651864066568649, 0.634084192553475,
        0.616688494561969, 0.599672088669339, 0.583030020204371, 0.5667572718654,
        0.550848768322639, 0.535299383967892, 0.520103947257001, 0.505257246260431,
        0.490754031684977, 0.476589022213249, 0.46275690208242, 0.449252325341552,
        0.436069912245555, 0.423205974605747, 0.4106531652521, 0.39840668039948,
        0.386461380891047, 0.374811984314975, 0.363453224264704, 0.352379851902848,
        0.341586644916259, 0.331068403184832, 0.320819956874279, 0.31083616902857,
        0.301110995958752, 0.291641183389757, 0.282420831386121, 0.273444955588216,
        0.264708614833586, 0.256206914916444, 0.247935008593902, 0.239888099677924,
        0.232061441819675, 0.224450342118235, 0.217050162160312, 0.209856317524031,
        0.202864281204524, 0.196069583611474, 0.189467814398248, 0.183054621446351,
        0.176825713015038, 0.17077685928726, 0.164903890637922, 0.159202699934773,
        0.153669242222215, 0.148299535941784, 0.143089661250278, 0.138035764053223,
        0.133134049958711, 0.12838079222654, 0.123772324007265, 0.119305671122251,
        0.114976081494676};
const double UMAP_B[101] = {
        0.790494973419029, 0.80063784415826, 0.810876441425738, 0.821199202674006,
        0.831595366275022, 0.84205539236769, 0.852571713401325, 0.863135518043442,
        0.873741680140683, 0.884384956993888, 0.895060878257082, 0.905765637284042,
        0.916495998501859, 0.927249214280422, 0.938022954467018, 0.948815759038301,
        0.95962499558526, 0.970449732070657, 0.981288783823989, 0.992141168965973,
        1.00300608092206, 1.01388286515112, 1.02477099750548, 1.03567006898871,
        1.04657977025277, 1.05749987674998, 1.06843023939592, 1.07937077470387,
        1.09032145585694, 1.10128169075827, 1.11225322117536, 1.12323470900213,
        1.13422639755358, 1.14522861434516, 1.15624176559097, 1.16726633179917,
        1.17830241385901, 1.18934945144456, 1.20040819996369, 1.21147891097075,
        1.22256381651844, 1.23366041866219, 1.24477022428392, 1.2558936051142,
        1.26703094885274, 1.27818265467871, 1.28934756395537, 1.30052872175886,
        1.31172539107843, 1.32293800168803, 1.3341669930459, 1.34541281413396,
        1.35667592718974, 1.36795680610473, 1.37925594017143, 1.39057383474783,
        1.40191101858967, 1.41326804557094, 1.42464550789942, 1.436044048272,
        1.44746436980037, 1.45890393087319, 1.47036701291879, 1.48185337703821,
        1.49336326709497, 1.50489726618312, 1.51645596605121, 1.52803997486173,
        1.53964990048402, 1.55128637349183, 1.56295003156298, 1.57464152150044,
        1.58636409305622, 1.59811350189048, 1.60989278253114, 1.62170263415549,
        1.63354377154668, 1.64541692037945, 1.65732282325244, 1.66926223230814,
        1.68123591907029, 1.69324466615879, 1.70528927262371, 1.71737055545595,
        1.72948934595558, 1.74164649289645, 1.75384285823827, 1.76607932576738,
        1.77835679827623, 1.79067619009556, 1.80303844043406, 1.81544450541945,
        1.82789536263139, 1.84039200538657, 1.85293545544251, 1.86552674229068,
        1.87816693701183, 1.89085711093115, 1.90359837758981, 1.91638829237987,
        1.92923479503841};

arma::mat transform_layout(arma::sp_mat &G, arma::mat &reference_layout, bool presmooth_network,
                           const std::string &method, double min_dist, double spread,
                           double gamma, unsigned int n_epochs, int thread_no,
                           int seed, double learning_rate, int sim2dist) {
    arma::mat coordinates;
    if (thread_no <= 0) {
        thread_no = SYS_THREADS_DEF;
    }

    auto found = find_ab(spread, min_dist);
    double a = found.first;
    double b = found.second;

    // a = UMAP_A[50];
    // b = UMAP_B[50];

    if (reference_layout.n_rows != G.n_cols) {
        stderr_printf(
                "Number of rows in the reference_layout should match with the number "
                "of columns in G\n");
        FLUSH;
        return (coordinates);
    }

    int Nq = G.n_rows, Nr = G.n_cols, D = reference_layout.n_cols;
    stdout_printf(
            "Transforming graph G with %d vertices, using a reference of %d "
            "vertices, in a %dD dimensions (%d threads)\n",
            Nq, Nr, D, thread_no);
    stdout_printf("\tmethod = %s, a = %.3f, b = %.3f (epochs = %d, threads=%d)\n",
                  method.c_str(), a, b, n_epochs, thread_no);

    arma::sp_mat W = normalizeGraph(G, 1);
    arma::mat query_layout = spmat_mat_product_parallel(W, reference_layout, thread_no);

    bool move_other = false;
    std::size_t grain_size = 1;
    bool pcg_rand = true;
    bool approx_pow = true;
    bool batch = true;
    std::string opt_name = "adam";
    double alpha = ADAM_ALPHA, beta1 = ADAM_BETA1, beta2 = ADAM_BETA2,
            eps = ADAM_EPS, negative_sample_rate = NEGATIVE_SAMPLE_RATE;

    arma::field<arma::mat> res(3);
    std::mt19937_64 engine(seed);

    // Encode positive edges of the graph
    arma::sp_mat H = G;

    double w_max = max(max(H));
    H.clean(w_max / n_epochs);

    arma::sp_mat Ht = trans(H);
    Ht.sync();

    unsigned int nE = H.n_nonzero;
    std::vector<unsigned int> positive_head(nE);
    std::vector<unsigned int> positive_tail(nE);
    std::vector<float> epochs_per_sample(nE);

    std::vector<unsigned int> positive_ptr(Ht.n_cols + 1);

    int i = 0;
    if (batch == false) {
        for (arma::sp_mat::iterator it = H.begin(); it != H.end(); ++it) {
            epochs_per_sample[i] = w_max / (*it);
            positive_head[i] = it.row();
            positive_tail[i] = it.col();
            i++;
        }
    } else {
        for (arma::sp_mat::iterator it = Ht.begin(); it != Ht.end(); ++it) {
            epochs_per_sample[i] = w_max / (*it);
            positive_tail[i] = it.row();
            positive_head[i] = it.col();
            i++;
        }
        for (int k = 0; k < Ht.n_cols + 1; k++) {
            positive_ptr[k] = Ht.col_ptrs[k];
        }
    }

    query_layout = arma::trans(query_layout);
    reference_layout = arma::trans(reference_layout);

    // Initial coordinates of vertices (0-simplices)
    std::vector<float> head_embedding(query_layout.n_elem);
    arma::fmat sub_coor = arma::conv_to<arma::fmat>::from(query_layout);
    memcpy(head_embedding.data(), sub_coor.memptr(),
           sizeof(float) * head_embedding.size());
    std::vector<float> tail_embedding(reference_layout.n_elem);
    sub_coor = arma::conv_to<arma::fmat>::from(reference_layout);
    memcpy(tail_embedding.data(), sub_coor.memptr(),
           sizeof(float) * tail_embedding.size());
    uwot::Coords coords = uwot::Coords(head_embedding, tail_embedding);

    UmapFactory umap_factory(
            move_other, pcg_rand, coords.get_head_embedding(),
            coords.get_tail_embedding(), positive_head, positive_tail, positive_ptr,
            n_epochs, Nq, Nr, epochs_per_sample, learning_rate, negative_sample_rate,
            batch, thread_no, grain_size, opt_name, alpha, beta1, beta2, eps, engine);

    stdout_printf("Transforming layout ... ");
    FLUSH;
    if (method == "umap") {
        create_umap(umap_factory, a, b, gamma, approx_pow);
    } else if (method == "tumap") {
        create_tumap(umap_factory);
    } else if (method == "largevis") {
        create_largevis(umap_factory, gamma);
    } else if (method == "pacmap") {
        create_pacmap(umap_factory, a, b);
    } else {
        stderr_printf("Unknown method: %s\n", method.c_str());
        FLUSH;
        return (coordinates);
    }
    arma::fmat coordinates_float(coords.get_head_embedding().data(), 2, Nq);
    coordinates = arma::trans(arma::conv_to<arma::mat>::from(coordinates_float));
    stdout_printf("done\n");
    FLUSH;

    return (coordinates);
}

arma::sp_mat smoothKNN(arma::sp_mat &D, int max_iter, double epsilon, double bandwidth, double local_connectivity,
                       double min_k_dist_scale, double min_sim, int thread_no) {
    int nV = D.n_cols;
    arma::sp_mat G = D;

    //#pragma omp parallel for num_threads(thread_no)
    for (int i = 0; i < nV; i++) {
        //  ParallelFor(0, nV, thread_no, [&](size_t i, size_t threadId) {
        arma::sp_mat v = D.col(i);
        arma::vec vals = nonzeros(v);
        if (vals.n_elem > local_connectivity) {
            double rho = min(vals);
            arma::vec negated_shifted_vals = -(vals - rho);
            arma::uvec deflate = arma::find(vals <= rho);
            negated_shifted_vals(deflate).zeros();

            double target = std::log2(vals.n_elem + 1);

            // Binary search to find optimal sigma
            double sigma = 1.0;
            double lo = 0.0;
            auto hi = DBL_MAX;

            int j;
            for (j = 0; j < max_iter; j++) {
                double obj = sum(exp(negated_shifted_vals / sigma));

                if (abs(obj - target) < epsilon) {
                    break;
                }

                if (target < obj) {
                    hi = sigma;
                    sigma = 0.5 * (lo + hi);
                } else {
                    lo = sigma;
                    if (hi == DBL_MAX) {
                        sigma *= 2;
                    } else {
                        sigma = 0.5 * (lo + hi);
                    }
                }
            }

            // double obj = sum(exp(negated_shifted_vals / sigma));
            // TODO: This is obviously a bug. `mean_dist` does not exist.
            double mean_dist = arma::mean(mean_dist);
            sigma = std::max(min_k_dist_scale * mean_dist, sigma);

            for (arma::sp_mat::col_iterator it = G.begin_col(i); it != G.end_col(i); ++it) {
                *it = std::max(min_sim, std::exp(-std::max(0.0, (*it) - rho) / (sigma * bandwidth)));
            }
        } else {
            for (arma::sp_mat::col_iterator it = G.begin_col(i); it != G.end_col(i); ++it) {
                *it = 1.0;
            }
        }
    }

    return (G);
}