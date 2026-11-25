#include "hist_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef ADASIGBOOST_USE_OPENMP
#include <omp.h>
#endif

namespace adasigboost {

std::vector<double> TreeModel::predict(const double* X, std::size_t n_samples) const {
    std::vector<double> out(n_samples, 0.0);
    if (feature.empty()) return out;

#ifdef ADASIGBOOST_USE_OPENMP
#pragma omp parallel for schedule(static) if (n_samples > 1024)
#endif
    for (std::size_t i = 0; i < n_samples; ++i) {
        const double* xi = X + i * static_cast<std::size_t>(n_features);
        int node = 0;
        while (feature[node] >= 0) {
            const int f = feature[node];
            const double val = xi[f];
            if (val <= threshold[node] || std::isnan(val)) {
                node = left_child[node];
            } else {
                node = right_child[node];
            }
        }
        out[i] = leaf_value[node];
    }
    return out;
}

BinningSpec compute_quantile_bins(
    const double* X,
    std::size_t n_samples,
    std::size_t n_features,
    int max_bins
) {
    if (max_bins < 2 || max_bins > 256) {
        throw std::invalid_argument("max_bins must be in [2, 256].");
    }
    BinningSpec spec;
    spec.n_features = static_cast<int>(n_features);
    spec.max_bins = max_bins;
    spec.bin_edges.resize(n_features);
    spec.X_binned.resize(n_samples * n_features, 0);

    if (n_samples == 0 || n_features == 0) {
        return spec;
    }

#ifdef ADASIGBOOST_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (n_features > 1)
#endif
    for (std::size_t f = 0; f < n_features; ++f) {
        std::vector<double> col;
        col.reserve(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i) {
            const double v = X[i * n_features + f];
            if (!std::isnan(v)) {
                col.push_back(v);
            }
        }
        if (col.empty()) continue;
        std::sort(col.begin(), col.end());

        std::vector<double> edges;
        edges.reserve(static_cast<std::size_t>(max_bins - 1));
        const std::size_t m = col.size();
        for (int b = 1; b < max_bins; ++b) {
            const std::size_t idx = (static_cast<std::size_t>(b) * m) / static_cast<std::size_t>(max_bins);
            const std::size_t clamped = std::min(idx, m - 1);
            const double val = col[clamped];
            if (edges.empty() || val > edges.back()) {
                edges.push_back(val);
            }
        }
        if (!edges.empty() && edges.back() >= col.back()) {
            edges.pop_back();
        }
        spec.bin_edges[f] = std::move(edges);
    }

#ifdef ADASIGBOOST_USE_OPENMP
#pragma omp parallel for schedule(static) if (n_samples > 1024)
#endif
    for (std::size_t i = 0; i < n_samples; ++i) {
        const double* xi = X + i * n_features;
        uint8_t* bi = spec.X_binned.data() + i * n_features;
        for (std::size_t f = 0; f < n_features; ++f) {
            const auto& edges = spec.bin_edges[f];
            if (edges.empty() || std::isnan(xi[f])) {
                bi[f] = 0;
            } else {
                auto it = std::lower_bound(edges.begin(), edges.end(), xi[f]);
                bi[f] = static_cast<uint8_t>(std::distance(edges.begin(), it));
            }
        }
    }

    return spec;
}

namespace {

struct BestSplit {
    int feature = -1;
    int bin = -1;
    double threshold = 0.0;
    double gain = -std::numeric_limits<double>::infinity();
    std::size_t left_count = 0;
};

struct NodeHist {
    std::vector<std::vector<double>> G;
    std::vector<std::vector<double>> H;
    std::vector<std::vector<std::size_t>> C;
};

NodeHist build_node_hist(
    const uint8_t* X_binned,
    const std::vector<std::vector<double>>& bin_edges,
    const double* g,
    const double* h,
    const std::vector<std::size_t>& indices,
    std::size_t start,
    std::size_t end,
    std::size_t n_features
) {
    NodeHist hist;
    hist.G.resize(n_features);
    hist.H.resize(n_features);
    hist.C.resize(n_features);
    for (std::size_t f = 0; f < n_features; ++f) {
        const std::size_t n_bins = bin_edges[f].size() + 1;
        hist.G[f].assign(n_bins, 0.0);
        hist.H[f].assign(n_bins, 0.0);
        hist.C[f].assign(n_bins, 0);
    }

#ifdef ADASIGBOOST_USE_OPENMP
#pragma omp parallel for schedule(dynamic) if (n_features > 1 && (end - start) > 256)
#endif
    for (std::size_t f = 0; f < n_features; ++f) {
        if (bin_edges[f].empty()) continue;
        auto& gf = hist.G[f];
        auto& hf = hist.H[f];
        auto& cf = hist.C[f];
        for (std::size_t pos = start; pos < end; ++pos) {
            const std::size_t idx = indices[pos];
            const uint8_t b = X_binned[idx * n_features + f];
            gf[b] += g[idx];
            hf[b] += h[idx];
            cf[b] += 1;
        }
    }
    return hist;
}

NodeHist subtract_hist(const NodeHist& parent, const NodeHist& child) {
    const std::size_t n_features = parent.G.size();
    NodeHist sibling;
    sibling.G.resize(n_features);
    sibling.H.resize(n_features);
    sibling.C.resize(n_features);
    for (std::size_t f = 0; f < n_features; ++f) {
        const std::size_t n_bins = parent.G[f].size();
        sibling.G[f].resize(n_bins);
        sibling.H[f].resize(n_bins);
        sibling.C[f].resize(n_bins);
        for (std::size_t b = 0; b < n_bins; ++b) {
            sibling.G[f][b] = parent.G[f][b] - child.G[f][b];
            sibling.H[f][b] = parent.H[f][b] - child.H[f][b];
            sibling.C[f][b] = parent.C[f][b] - child.C[f][b];
        }
    }
    return sibling;
}

int build_node(
    TreeModel& model,
    const uint8_t* X_binned,
    const std::vector<std::vector<double>>& bin_edges,
    const double* g,
    const double* h,
    std::vector<std::size_t>& indices,
    std::size_t start,
    std::size_t end,
    std::size_t n_features,
    int depth,
    int max_depth,
    double reg_lambda,
    double reg_gamma,
    double min_child_hessian,
    int min_samples_leaf,
    const NodeHist& hist,
    double G_total,
    double H_total
) {
    const int node_id = static_cast<int>(model.feature.size());
    model.feature.push_back(-1);
    model.threshold.push_back(0.0);
    model.left_child.push_back(-1);
    model.right_child.push_back(-1);
    model.leaf_value.push_back(0.0);

    const double leaf_val = -G_total / (H_total + reg_lambda);
    const std::size_t n_node = end - start;

    if (depth >= max_depth || n_node < static_cast<std::size_t>(2 * min_samples_leaf) ||
        H_total < 2.0 * min_child_hessian) {
        model.leaf_value[node_id] = leaf_val;
        model.n_leaves += 1;
        model.leaf_l2_sum += leaf_val * leaf_val;
        return node_id;
    }

    const double parent_score = (G_total * G_total) / (H_total + reg_lambda);
    std::vector<BestSplit> feature_splits(n_features);

    for (std::size_t f = 0; f < n_features; ++f) {
        const auto& edges = bin_edges[f];
        if (edges.empty()) continue;
        const int n_bins = static_cast<int>(edges.size()) + 1;
        const auto& bin_G = hist.G[f];
        const auto& bin_H = hist.H[f];
        const auto& bin_C = hist.C[f];

        double G_L = 0.0;
        double H_L = 0.0;
        std::size_t C_L = 0;
        BestSplit best_f;

        for (int b = 0; b < n_bins - 1; ++b) {
            G_L += bin_G[static_cast<std::size_t>(b)];
            H_L += bin_H[static_cast<std::size_t>(b)];
            C_L += bin_C[static_cast<std::size_t>(b)];

            const std::size_t C_R = n_node - C_L;
            if (C_L < static_cast<std::size_t>(min_samples_leaf) ||
                C_R < static_cast<std::size_t>(min_samples_leaf)) {
                continue;
            }
            const double G_R = G_total - G_L;
            const double H_R = H_total - H_L;
            if (H_L < min_child_hessian || H_R < min_child_hessian) {
                continue;
            }

            const double score_L = (G_L * G_L) / (H_L + reg_lambda);
            const double score_R = (G_R * G_R) / (H_R + reg_lambda);
            const double gain = 0.5 * (score_L + score_R - parent_score) - reg_gamma;

            if (gain > best_f.gain) {
                best_f.feature = static_cast<int>(f);
                best_f.bin = b;
                best_f.threshold = edges[static_cast<std::size_t>(b)];
                best_f.gain = gain;
                best_f.left_count = C_L;
            }
        }
        feature_splits[f] = best_f;
    }

    BestSplit global_best;
    for (std::size_t f = 0; f < n_features; ++f) {
        if (feature_splits[f].feature >= 0 && feature_splits[f].gain > global_best.gain) {
            global_best = feature_splits[f];
        }
    }

    if (global_best.feature < 0 || global_best.gain <= 0.0) {
        model.leaf_value[node_id] = leaf_val;
        model.n_leaves += 1;
        model.leaf_l2_sum += leaf_val * leaf_val;
        return node_id;
    }

    const int best_f = global_best.feature;
    const uint8_t best_b = static_cast<uint8_t>(global_best.bin);
    auto mid_it = std::partition(
        indices.begin() + static_cast<std::ptrdiff_t>(start),
        indices.begin() + static_cast<std::ptrdiff_t>(end),
        [&](std::size_t idx) {
            return X_binned[idx * n_features + static_cast<std::size_t>(best_f)] <= best_b;
        }
    );
    const std::size_t mid = static_cast<std::size_t>(std::distance(indices.begin(), mid_it));

    // Smaller-child histogram subtraction
    double G_L_tot = 0.0;
    double H_L_tot = 0.0;
    for (int b = 0; b <= global_best.bin; ++b) {
        G_L_tot += hist.G[static_cast<std::size_t>(best_f)][static_cast<std::size_t>(b)];
        H_L_tot += hist.H[static_cast<std::size_t>(best_f)][static_cast<std::size_t>(b)];
    }
    const double G_R_tot = G_total - G_L_tot;
    const double H_R_tot = H_total - H_L_tot;

    NodeHist left_hist;
    NodeHist right_hist;
    if ((mid - start) <= (end - mid)) {
        left_hist = build_node_hist(X_binned, bin_edges, g, h, indices, start, mid, n_features);
        right_hist = subtract_hist(hist, left_hist);
    } else {
        right_hist = build_node_hist(X_binned, bin_edges, g, h, indices, mid, end, n_features);
        left_hist = subtract_hist(hist, right_hist);
    }

    model.feature[node_id] = best_f;
    model.threshold[node_id] = global_best.threshold;

    const int left_id = build_node(
        model, X_binned, bin_edges, g, h, indices, start, mid, n_features,
        depth + 1, max_depth, reg_lambda, reg_gamma, min_child_hessian, min_samples_leaf,
        left_hist, G_L_tot, H_L_tot
    );
    const int right_id = build_node(
        model, X_binned, bin_edges, g, h, indices, mid, end, n_features,
        depth + 1, max_depth, reg_lambda, reg_gamma, min_child_hessian, min_samples_leaf,
        right_hist, G_R_tot, H_R_tot
    );

    model.left_child[node_id] = left_id;
    model.right_child[node_id] = right_id;
    return node_id;
}

}  // namespace

TreeModel fit_hist_tree(
    const uint8_t* X_binned,
    const std::vector<std::vector<double>>& bin_edges,
    const double* g,
    const double* h,
    std::size_t n_samples,
    std::size_t n_features,
    int max_depth,
    double reg_lambda,
    double reg_gamma,
    double min_child_hessian,
    int min_samples_leaf
) {
    if (reg_lambda <= 0.0) {
        throw std::invalid_argument("reg_lambda must be strictly positive.");
    }
    TreeModel model;
    model.n_features = static_cast<int>(n_features);
    model.max_depth = max_depth;
    model.reg_lambda = reg_lambda;
    model.reg_gamma = reg_gamma;

    if (n_samples == 0 || n_features == 0) {
        model.feature.push_back(-1);
        model.threshold.push_back(0.0);
        model.left_child.push_back(-1);
        model.right_child.push_back(-1);
        model.leaf_value.push_back(0.0);
        model.n_leaves = 1;
        return model;
    }

    std::vector<std::size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    double G_total = 0.0;
    double H_total = 0.0;
    for (std::size_t i = 0; i < n_samples; ++i) {
        G_total += g[i];
        H_total += h[i];
    }

    NodeHist root_hist = build_node_hist(
        X_binned, bin_edges, g, h, indices, 0, n_samples, n_features
    );

    build_node(
        model, X_binned, bin_edges, g, h, indices, 0, n_samples, n_features,
        0, max_depth, reg_lambda, reg_gamma, min_child_hessian, min_samples_leaf,
        root_hist, G_total, H_total
    );
    return model;
}

}  // namespace adasigboost
