#pragma once

#include <cstdint>
#include <vector>

namespace adasigboost {

struct TreeModel {
    int n_features = 0;
    int max_depth = 4;
    int n_leaves = 0;
    double reg_lambda = 1.0;
    double reg_gamma = 0.0;
    double leaf_l2_sum = 0.0;

    std::vector<int> feature;
    std::vector<double> threshold;
    std::vector<int> left_child;
    std::vector<int> right_child;
    std::vector<double> leaf_value;

    std::vector<double> predict(const double* X, std::size_t n_samples) const;
};

struct BinningSpec {
    int n_features = 0;
    int max_bins = 256;
    std::vector<std::vector<double>> bin_edges;
    std::vector<uint8_t> X_binned;
};

BinningSpec compute_quantile_bins(
    const double* X,
    std::size_t n_samples,
    std::size_t n_features,
    int max_bins = 256
);

TreeModel fit_hist_tree(
    const uint8_t* X_binned,
    const std::vector<std::vector<double>>& bin_edges,
    const double* g,
    const double* h,
    std::size_t n_samples,
    std::size_t n_features,
    int max_depth = 4,
    double reg_lambda = 1.0,
    double reg_gamma = 0.0,
    double min_child_hessian = 1e-3,
    int min_samples_leaf = 5
);

}  // namespace adasigboost
