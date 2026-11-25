#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "coord_descent.hpp"
#include "hist_tree.hpp"

namespace py = pybind11;

namespace {

py::array_t<double> tree_predict(
    const adasigboost::TreeModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> X
) {
    auto buf = X.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("X must be 2D.");
    }
    const auto n_samples = static_cast<std::size_t>(buf.shape[0]);
    const auto n_features = static_cast<std::size_t>(buf.shape[1]);
    if (static_cast<int>(n_features) != model.n_features) {
        throw std::invalid_argument("Feature dimension mismatch.");
    }
    const auto preds = model.predict(static_cast<const double*>(buf.ptr), n_samples);
    py::array_t<double> out(static_cast<py::ssize_t>(n_samples));
    auto out_buf = out.request();
    std::copy(preds.begin(), preds.end(), static_cast<double*>(out_buf.ptr));
    return out;
}

py::tuple py_compute_quantile_bins(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    int max_bins
) {
    auto buf = X.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("X must be 2D.");
    }
    const auto n_samples = static_cast<std::size_t>(buf.shape[0]);
    const auto n_features = static_cast<std::size_t>(buf.shape[1]);
    auto spec = adasigboost::compute_quantile_bins(
        static_cast<const double*>(buf.ptr), n_samples, n_features, max_bins
    );

    py::array_t<uint8_t> X_binned({static_cast<py::ssize_t>(n_samples), static_cast<py::ssize_t>(n_features)});
    auto b_buf = X_binned.request();
    std::copy(spec.X_binned.begin(), spec.X_binned.end(), static_cast<uint8_t*>(b_buf.ptr));

    return py::make_tuple(X_binned, spec.bin_edges);
}

adasigboost::TreeModel py_fit_hist_tree(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X_binned,
    const std::vector<std::vector<double>>& bin_edges,
    py::array_t<double, py::array::c_style | py::array::forcecast> g,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    int max_depth,
    double reg_lambda,
    double reg_gamma,
    double min_child_hessian,
    int min_samples_leaf
) {
    auto x_buf = X_binned.request();
    auto g_buf = g.request();
    auto h_buf = h.request();
    if (x_buf.ndim != 2 || g_buf.ndim != 1 || h_buf.ndim != 1) {
        throw std::invalid_argument("Expected 2D X_binned and 1D g, h.");
    }
    const auto n_samples = static_cast<std::size_t>(x_buf.shape[0]);
    const auto n_features = static_cast<std::size_t>(x_buf.shape[1]);
    if (static_cast<std::size_t>(g_buf.shape[0]) != n_samples ||
        static_cast<std::size_t>(h_buf.shape[0]) != n_samples ||
        bin_edges.size() != n_features) {
        throw std::invalid_argument("Input shape mismatch in fit_hist_tree.");
    }
    return adasigboost::fit_hist_tree(
        static_cast<const uint8_t*>(x_buf.ptr),
        bin_edges,
        static_cast<const double*>(g_buf.ptr),
        static_cast<const double*>(h_buf.ptr),
        n_samples,
        n_features,
        max_depth,
        reg_lambda,
        reg_gamma,
        min_child_hessian,
        min_samples_leaf
    );
}

py::tuple py_solve_elastic_net_weights(
    py::array_t<double, py::array::c_style | py::array::forcecast> Z,
    py::array_t<double, py::array::c_style | py::array::forcecast> g,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> w_init,
    double lambda1,
    double lambda2,
    int max_iter,
    double tol
) {
    auto z_buf = Z.request();
    auto g_buf = g.request();
    auto h_buf = h.request();
    auto w_buf = w_init.request();

    if (z_buf.ndim != 2 || g_buf.ndim != 1 || h_buf.ndim != 1 || w_buf.ndim != 1) {
        throw std::invalid_argument("Invalid dimensions for solve_elastic_net_weights.");
    }
    const auto n_samples = static_cast<std::size_t>(z_buf.shape[0]);
    const auto n_modes = static_cast<std::size_t>(z_buf.shape[1]);
    if (static_cast<std::size_t>(g_buf.shape[0]) != n_samples ||
        static_cast<std::size_t>(h_buf.shape[0]) != n_samples ||
        static_cast<std::size_t>(w_buf.shape[0]) != n_modes) {
        throw std::invalid_argument("Shape mismatch in solve_elastic_net_weights.");
    }

    auto res = adasigboost::solve_elastic_net_weights(
        static_cast<const double*>(z_buf.ptr),
        static_cast<const double*>(g_buf.ptr),
        static_cast<const double*>(h_buf.ptr),
        static_cast<const double*>(w_buf.ptr),
        n_samples,
        n_modes,
        lambda1,
        lambda2,
        max_iter,
        tol
    );

    py::array_t<double> w_out(static_cast<py::ssize_t>(n_modes));
    auto out_buf = w_out.request();
    std::copy(res.weights.begin(), res.weights.end(), static_cast<double*>(out_buf.ptr));
    return py::make_tuple(w_out, res.iterations, res.max_delta);
}

py::tuple py_solve_group_elastic_net_weights(
    py::array_t<double, py::array::c_style | py::array::forcecast> Z,
    py::array_t<double, py::array::c_style | py::array::forcecast> g,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> w_init,
    double lambda1,
    double lambda2,
    int max_iter,
    double tol
) {
    auto z_buf = Z.request();
    auto g_buf = g.request();
    auto h_buf = h.request();
    auto w_buf = w_init.request();

    if (z_buf.ndim != 3 || g_buf.ndim != 2 || h_buf.ndim != 2 || w_buf.ndim != 2) {
        throw std::invalid_argument("Expected 3D Z, 2D g, 2D h, 2D w_init.");
    }
    const auto n_samples = static_cast<std::size_t>(z_buf.shape[0]);
    const auto n_modes = static_cast<std::size_t>(z_buf.shape[1]);
    const auto n_outputs = static_cast<std::size_t>(z_buf.shape[2]);

    if (static_cast<std::size_t>(g_buf.shape[0]) != n_samples ||
        static_cast<std::size_t>(g_buf.shape[1]) != n_outputs ||
        static_cast<std::size_t>(h_buf.shape[0]) != n_samples ||
        static_cast<std::size_t>(h_buf.shape[1]) != n_outputs ||
        static_cast<std::size_t>(w_buf.shape[0]) != n_modes ||
        static_cast<std::size_t>(w_buf.shape[1]) != n_outputs) {
        throw std::invalid_argument("Shape mismatch in solve_group_elastic_net_weights.");
    }

    auto res = adasigboost::solve_group_elastic_net_weights(
        static_cast<const double*>(z_buf.ptr),
        static_cast<const double*>(g_buf.ptr),
        static_cast<const double*>(h_buf.ptr),
        static_cast<const double*>(w_buf.ptr),
        n_samples,
        n_modes,
        n_outputs,
        lambda1,
        lambda2,
        max_iter,
        tol
    );

    py::array_t<double> w_out({static_cast<py::ssize_t>(n_modes), static_cast<py::ssize_t>(n_outputs)});
    auto out_buf = w_out.request();
    std::copy(res.weights.begin(), res.weights.end(), static_cast<double*>(out_buf.ptr));
    return py::make_tuple(w_out, res.iterations, res.max_delta);
}

}  // namespace

PYBIND11_MODULE(_adasigboost_cpp, m) {
    m.doc() = "AdaSigBoost C++17 / OpenMP accelerated tree and coordinate descent backend";

    py::class_<adasigboost::TreeModel>(m, "TreeModel")
        .def_readonly("n_features", &adasigboost::TreeModel::n_features)
        .def_readonly("max_depth", &adasigboost::TreeModel::max_depth)
        .def_readonly("n_leaves", &adasigboost::TreeModel::n_leaves)
        .def_readonly("reg_lambda", &adasigboost::TreeModel::reg_lambda)
        .def_readonly("reg_gamma", &adasigboost::TreeModel::reg_gamma)
        .def_readonly("leaf_l2_sum", &adasigboost::TreeModel::leaf_l2_sum)
        .def_readonly("feature", &adasigboost::TreeModel::feature)
        .def_readonly("threshold", &adasigboost::TreeModel::threshold)
        .def_readonly("left_child", &adasigboost::TreeModel::left_child)
        .def_readonly("right_child", &adasigboost::TreeModel::right_child)
        .def_readonly("leaf_value", &adasigboost::TreeModel::leaf_value)
        .def("predict", &tree_predict, py::arg("X"))
        .def(py::pickle(
            [](const adasigboost::TreeModel& t) {
                return py::make_tuple(
                    t.n_features, t.max_depth, t.n_leaves, t.reg_lambda, t.reg_gamma,
                    t.leaf_l2_sum, t.feature, t.threshold, t.left_child, t.right_child, t.leaf_value
                );
            },
            [](py::tuple tup) {
                if (tup.size() != 11) throw std::runtime_error("Invalid TreeModel pickle state.");
                adasigboost::TreeModel t;
                t.n_features = tup[0].cast<int>();
                t.max_depth = tup[1].cast<int>();
                t.n_leaves = tup[2].cast<int>();
                t.reg_lambda = tup[3].cast<double>();
                t.reg_gamma = tup[4].cast<double>();
                t.leaf_l2_sum = tup[5].cast<double>();
                t.feature = tup[6].cast<std::vector<int>>();
                t.threshold = tup[7].cast<std::vector<double>>();
                t.left_child = tup[8].cast<std::vector<int>>();
                t.right_child = tup[9].cast<std::vector<int>>();
                t.leaf_value = tup[10].cast<std::vector<double>>();
                return t;
            }
        ));

    m.def("compute_quantile_bins", &py_compute_quantile_bins, py::arg("X"), py::arg("max_bins") = 256);
    m.def(
        "fit_hist_tree",
        &py_fit_hist_tree,
        py::arg("X_binned"),
        py::arg("bin_edges"),
        py::arg("g"),
        py::arg("h"),
        py::arg("max_depth") = 4,
        py::arg("reg_lambda") = 1.0,
        py::arg("reg_gamma") = 0.0,
        py::arg("min_child_hessian") = 1e-3,
        py::arg("min_samples_leaf") = 5
    );
    m.def(
        "solve_elastic_net_weights",
        &py_solve_elastic_net_weights,
        py::arg("Z"),
        py::arg("g"),
        py::arg("h"),
        py::arg("w_init"),
        py::arg("lambda1"),
        py::arg("lambda2"),
        py::arg("max_iter") = 500,
        py::arg("tol") = 1e-7
    );
    m.def(
        "solve_group_elastic_net_weights",
        &py_solve_group_elastic_net_weights,
        py::arg("Z"),
        py::arg("g"),
        py::arg("h"),
        py::arg("w_init"),
        py::arg("lambda1"),
        py::arg("lambda2"),
        py::arg("max_iter") = 500,
        py::arg("tol") = 1e-7
    );
}
