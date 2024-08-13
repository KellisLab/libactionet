#ifndef LIBACTIONET_FIND_AB_HPP
#define LIBACTIONET_FIND_AB_HPP

#include <vector>
#include <utility>

// From umappp "https://github.com/LTLA/umappp/blob/master/include/umappp/find_ab.hpp"
template<typename Float>
std::pair<Float, Float>
find_ab(Float spread, Float min_dist, Float grid = 300, Float limit = 0.5, int iter = 50, Float tol = 1e-6) {
    Float x_half = std::log(limit) * -spread + min_dist;
    Float d_half = limit / -spread;

    // Compute the x and y coordinates of the expected distance curve.
    std::vector<Float> grid_x(grid), grid_y(grid), log_x(grid);
    const Float delta = spread * 3 / grid;
    for (int g = 0; g < grid; ++g) {
        grid_x[g] =
                (g + 1) * delta; // +1 to avoid meaningless least squares result at x =
        // 0, where both curves have y = 1 (and also the
        // derivative w.r.t. b is not defined).
        log_x[g] = std::log(grid_x[g]);
        grid_y[g] =
                (grid_x[g] <= min_dist ? 1
                                       : std::exp(-(grid_x[g] - min_dist) / spread));
    }

    // Starting estimates.
    Float b = -d_half * x_half / (1 / limit - 1) / (2 * limit * limit);
    Float a = (1 / limit - 1) / std::pow(x_half, 2 * b);

    std::vector<Float> observed_y(grid), xpow(grid);
    auto compute_ss = [&](Float A, Float B) -> Float {
        for (int g = 0; g < grid; ++g) {
            xpow[g] = std::pow(grid_x[g], 2 * B);
            observed_y[g] = 1 / (1 + A * xpow[g]);
        }

        Float ss = 0;
        for (int g = 0; g < grid; ++g) {
            ss += (grid_y[g] - observed_y[g]) * (grid_y[g] - observed_y[g]);
        }

        return ss;
    };
    Float ss = compute_ss(a, b);

    for (int it = 0; it < iter; ++it) {
        // Computing the first and second derivatives of the sum of squared
        // differences.
        Float da = 0, db = 0, daa = 0, dab = 0, dbb = 0;
        for (int g = 0; g < grid; ++g) {
            const Float &x = grid_x[g];
            const Float &gy = grid_y[g];
            const Float &oy = observed_y[g];

            const Float &x2b = xpow[g];
            const Float logx2 = log_x[g] * 2;
            const Float delta = oy - gy;

            da += -2 * x2b * oy * oy * delta;

            db += -2 * a * x2b * logx2 * oy * oy * delta;

            daa += 2 * (x2b * oy * oy * x2b * oy * oy +
                        x2b * 2 * x2b * oy * oy * oy * delta);

            dab +=
                    -2 *
                    ((x2b * logx2 * oy * oy - a * x2b * logx2 * 2 * x2b * oy * oy * oy) *
                     delta -
                     a * x2b * logx2 * oy * oy * x2b * oy * oy);

            dbb += -2 * (((a * x2b * logx2 * logx2 * oy * oy) -
                          (a * x2b * logx2 * 2 * a * x2b * logx2 * oy * oy * oy)) *
                         delta -
                         a * x2b * logx2 * oy * oy * a * x2b * logx2 * oy * oy);
        }

        // Applying the Newton iterations with damping.
        Float determinant = daa * dbb - dab * dab;
        const Float delta_a = (da * dbb - dab * db) / determinant;
        const Float delta_b = (-da * dab + daa * db) / determinant;

        Float ss_next = 0;
        Float factor = 1;
        for (int inner = 0; inner < 10; ++inner, factor /= 2) {
            ss_next = compute_ss(a - factor * delta_a, b - factor * delta_b);
            if (ss_next < ss) {
                break;
            }
        }

        if (ss && 1 - ss_next / ss > tol) {
            a -= factor * delta_a;
            b -= factor * delta_b;
            ss = ss_next;
        } else {
            break;
        }
    }

    return std::make_pair(a, b);
}

#endif //LIBACTIONET_FIND_AB_HPP
