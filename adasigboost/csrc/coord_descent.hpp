#pragma once

#include <cstddef>
#include <vector>

namespace adasigboost {

struct CoordDescentResult {
    std::vector<double> weights;
    int iterations;
    double max_delta;
};

/**
 * Solve scalar Elastic-Net weighted least squares (L_out = 1):
 *   min_w sum_i [ g_i (Z_i^T w) + 0.5 * h_i (Z_i^T w)^2 ] + lambda1 * ||w||_1 + 0.5 * lambda2 * ||w||_2^2
 * Z is row-major (n_samples, n_modes).
 */
CoordDescentResult solve_elastic_net_weights(
    const double* Z,
    const double* g,
    const double* h,
    const double* w_init,
    std::size_t n_samples,
    std::size_t n_modes,
    double lambda1,
    double lambda2,
    int max_iter = 500,
    double tol = 1e-7
);

/**
 * Solve Group-Elastic-Net weighted least squares (L_out >= 1):
 *   min_W sum_{i,c} [ g_{i,c} (sum_k W_{k,c} Z_{i,k,c}) + 0.5 * h_{i,c} (sum_k W_{k,c} Z_{i,k,c})^2 ]
 *         + lambda1 * sum_k ||W_{k,:}||_2 + 0.5 * lambda2 * ||W||_F^2
 *
 * Z is row-major (n_samples, n_modes, n_outputs).
 * g, h are row-major (n_samples, n_outputs).
 * w_init is row-major (n_modes, n_outputs).
 */
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
    int max_iter = 500,
    double tol = 1e-7
);

}  // namespace adasigboost
