#include "coord_descent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace adasigboost {

namespace {
inline double soft_threshold(double rho, double lambda1) {
    if (rho > lambda1) return rho - lambda1;
    if (rho < -lambda1) return rho + lambda1;
    return 0.0;
}

/**
 * Solve sum_c rho_c^2 / (zeta_c * tau + lambda1)^2 = 1 for tau > 0
 * given ||rho||_2 > lambda1 and zeta_c > 0.
 */
double solve_group_secular(
    const double* rho,
    const double* zeta,
    std::size_t n_outputs,
    double lambda1
) {
    double rho_norm_sq = 0.0;
    double zeta_min = std::numeric_limits<double>::infinity();
    double zeta_max = 0.0;
    for (std::size_t c = 0; c < n_outputs; ++c) {
        rho_norm_sq += rho[c] * rho[c];
        zeta_min = std::min(zeta_min, zeta[c]);
        zeta_max = std::max(zeta_max, zeta[c]);
    }
    const double rho_norm = std::sqrt(rho_norm_sq);
    if (rho_norm <= lambda1 || rho_norm == 0.0) {
        return 0.0;
    }

    double lo = std::max(0.0, (rho_norm - lambda1) / zeta_max);
    double hi = std::max(lo + 1e-12, (rho_norm - lambda1) / zeta_min);
    double tau = 0.5 * (lo + hi);

    for (int it = 0; it < 60; ++it) {
        double f = -1.0;
        double df = 0.0;
        for (std::size_t c = 0; c < n_outputs; ++c) {
            const double denom = zeta[c] * tau + lambda1;
            const double r2 = rho[c] * rho[c];
            const double term = r2 / (denom * denom);
            f += term;
            df -= 2.0 * term * zeta[c] / denom;
        }
        if (std::abs(f) < 1e-12 || (hi - lo) < 1e-13 * (1.0 + tau)) {
            break;
        }
        if (f > 0.0) {
            lo = tau;
        } else {
            hi = tau;
        }
        double next_tau = (df < -1e-30) ? (tau - f / df) : 0.5 * (lo + hi);
        if (!(next_tau > lo && next_tau < hi)) {
            next_tau = 0.5 * (lo + hi);
        }
        tau = next_tau;
    }
    return tau;
}
}  // namespace

CoordDescentResult solve_elastic_net_weights(
    const double* Z,
    const double* g,
    const double* h,
    const double* w_init,
    std::size_t n_samples,
    std::size_t n_modes,
    double lambda1,
    double lambda2,
    int max_iter,
    double tol
) {
    if (lambda1 < 0.0 || lambda2 <= 0.0) {
        throw std::invalid_argument("Require lambda1 >= 0 and lambda2 > 0.");
    }
    std::vector<double> w(n_modes, 0.0);
    if (w_init != nullptr) {
        std::copy(w_init, w_init + n_modes, w.begin());
    }
    if (n_samples == 0 || n_modes == 0) {
        return {w, 0, 0.0};
    }

    std::vector<double> zeta(n_modes, lambda2);
    std::vector<double> p(n_samples, 0.0);

    for (std::size_t i = 0; i < n_samples; ++i) {
        const double hi = h[i];
        const double* zi = Z + i * n_modes;
        double pi = 0.0;
        for (std::size_t k = 0; k < n_modes; ++k) {
            const double zik = zi[k];
            zeta[k] += hi * zik * zik;
            pi += w[k] * zik;
        }
        p[i] = pi;
    }

    int it = 0;
    double max_delta = 0.0;
    for (; it < max_iter; ++it) {
        max_delta = 0.0;
        for (std::size_t k = 0; k < n_modes; ++k) {
            const double w_old = w[k];
            double rho_k = 0.0;
            for (std::size_t i = 0; i < n_samples; ++i) {
                const double zik = Z[i * n_modes + k];
                const double p_excl = p[i] - w_old * zik;
                rho_k -= zik * (g[i] + h[i] * p_excl);
            }
            const double w_new = soft_threshold(rho_k, lambda1) / zeta[k];
            const double diff = w_new - w_old;
            if (diff != 0.0) {
                w[k] = w_new;
                for (std::size_t i = 0; i < n_samples; ++i) {
                    p[i] += diff * Z[i * n_modes + k];
                }
            }
            max_delta = std::max(max_delta, std::abs(diff));
        }
        if (max_delta < tol) {
            ++it;
            break;
        }
    }

    return {w, it, max_delta};
}

CoordDescentResult solve_group_elastic_net_weights(
    const double* Z,
    const double* g,
    const double* h,
    const double* w_init,
    std::size_t n_samples,
    std::size_t n_modes,
    std::size_t n_outputs,
    double lambda1,
    double lambda2,
    int max_iter,
    double tol
) {
    if (lambda1 < 0.0 || lambda2 <= 0.0) {
        throw std::invalid_argument("Require lambda1 >= 0 and lambda2 > 0.");
    }
    if (n_outputs == 1) {
        return solve_elastic_net_weights(
            Z, g, h, w_init, n_samples, n_modes, lambda1, lambda2, max_iter, tol
        );
    }

    const std::size_t total_w = n_modes * n_outputs;
    std::vector<double> w(total_w, 0.0);
    if (w_init != nullptr) {
        std::copy(w_init, w_init + total_w, w.begin());
    }
    if (n_samples == 0 || n_modes == 0 || n_outputs == 0) {
        return {w, 0, 0.0};
    }

    // zeta[k * n_outputs + c] = sum_i h_{i,c} * Z_{i,k,c}^2 + lambda2
    std::vector<double> zeta(total_w, lambda2);
    // p[i * n_outputs + c] = sum_k w_{k,c} * Z_{i,k,c}
    std::vector<double> p(n_samples * n_outputs, 0.0);

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t k = 0; k < n_modes; ++k) {
            for (std::size_t c = 0; c < n_outputs; ++c) {
                const double zikc = Z[(i * n_modes + k) * n_outputs + c];
                const double hic = h[i * n_outputs + c];
                zeta[k * n_outputs + c] += hic * zikc * zikc;
                p[i * n_outputs + c] += w[k * n_outputs + c] * zikc;
            }
        }
    }

    std::vector<double> rho_k(n_outputs, 0.0);
    std::vector<double> w_new_k(n_outputs, 0.0);

    int it = 0;
    double max_delta = 0.0;
    for (; it < max_iter; ++it) {
        max_delta = 0.0;
        for (std::size_t k = 0; k < n_modes; ++k) {
            std::fill(rho_k.begin(), rho_k.end(), 0.0);
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t c = 0; c < n_outputs; ++c) {
                    const double zikc = Z[(i * n_modes + k) * n_outputs + c];
                    const double p_excl = p[i * n_outputs + c] - w[k * n_outputs + c] * zikc;
                    rho_k[c] -= zikc * (g[i * n_outputs + c] + h[i * n_outputs + c] * p_excl);
                }
            }

            double rho_sq = 0.0;
            for (std::size_t c = 0; c < n_outputs; ++c) {
                rho_sq += rho_k[c] * rho_k[c];
            }
            const double rho_norm = std::sqrt(rho_sq);

            if (rho_norm <= lambda1) {
                std::fill(w_new_k.begin(), w_new_k.end(), 0.0);
            } else if (lambda1 == 0.0) {
                for (std::size_t c = 0; c < n_outputs; ++c) {
                    w_new_k[c] = rho_k[c] / zeta[k * n_outputs + c];
                }
            } else {
                const double tau = solve_group_secular(
                    rho_k.data(), zeta.data() + k * n_outputs, n_outputs, lambda1
                );
                for (std::size_t c = 0; c < n_outputs; ++c) {
                    w_new_k[c] = rho_k[c] / (zeta[k * n_outputs + c] + lambda1 / tau);
                }
            }

            for (std::size_t c = 0; c < n_outputs; ++c) {
                const double diff = w_new_k[c] - w[k * n_outputs + c];
                if (diff != 0.0) {
                    w[k * n_outputs + c] = w_new_k[c];
                    for (std::size_t i = 0; i < n_samples; ++i) {
                        p[i * n_outputs + c] += diff * Z[(i * n_modes + k) * n_outputs + c];
                    }
                    max_delta = std::max(max_delta, std::abs(diff));
                }
            }
        }
        if (max_delta < tol) {
            ++it;
            break;
        }
    }

    return {w, it, max_delta};
}

}  // namespace adasigboost
