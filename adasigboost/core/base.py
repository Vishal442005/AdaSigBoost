"""Core AdaSigBoost training loop, OOF residual engine, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import validate_data

from .. import _adasigboost_cpp  # type: ignore[attr-defined]
from ..decomposition.base import FilterBankParams
from ..decomposition.filterbank import apply_filterbank_np, fit_stage_filterbank
from ..descriptors.time_freq import extract_descriptors
from ..utils.seeding import set_deterministic_seed
from ..utils.validation import check_signal_and_tabular


@dataclass
class StageModel:
    """Artifacts saved for a single boosting stage m."""

    filter_params: FilterBankParams
    mode_trees: list[list[Any | None]]  # shape [K][L_out] (or [1][L_out] in global_tree_mode)
    tabular_trees: list[Any] | None  # length L_out if Z was provided
    weights: np.ndarray  # shape (K, L_out)
    active_modes: list[int]
    global_tree_mode: bool = False


def _evaluate_tree_subobj(
    g: np.ndarray,
    h: np.ndarray,
    pred: np.ndarray,
    tree: Any,
) -> float:
    """Compute exact regularized tree sub-objective value."""
    quad = float(np.sum(g * pred + 0.5 * h * (pred * pred)))
    reg = float(tree.reg_gamma * tree.n_leaves + 0.5 * tree.reg_lambda * tree.leaf_l2_sum)
    return quad + reg


class AdaSigBoostBase(BaseEstimator):
    """Shared base class for AdaSigBoostClassifier and AdaSigBoostRegressor."""

    def __init__(
        self,
        n_estimators: int = 50,
        n_modes: int = 8,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        lambda1: float = 0.5,
        lambda2: float = 1.0,
        alpha1: float = 0.05,
        alpha2: float = 0.01,
        reg_gamma: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_hessian: float = 1e-3,
        min_samples_leaf: int = 5,
        max_bins: int = 256,
        n_inner_iters: int = 2,
        n_filter_steps: int = 16,
        filter_lr: float = 0.08,
        dictionary_type: str = "composite",
        oof_folds: int = 5,
        use_oof: bool = True,
        mode_dropout: float = 0.0,
        use_bic_stopping: bool = True,
        bic_scale: float = 0.05,
        bic_patience: int = 5,
        min_stages: int = 5,
        freeze_decomposition: bool = False,
        uniform_weights: bool = False,
        global_tree_mode: bool = False,
        block1_mode: str = "full",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_modes = n_modes
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.reg_gamma = reg_gamma
        self.reg_lambda = reg_lambda
        self.min_child_hessian = min_child_hessian
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.n_inner_iters = n_inner_iters
        self.n_filter_steps = n_filter_steps
        self.filter_lr = filter_lr
        self.dictionary_type = dictionary_type
        self.oof_folds = oof_folds
        self.use_oof = use_oof
        self.mode_dropout = mode_dropout
        self.use_bic_stopping = use_bic_stopping
        self.bic_scale = bic_scale
        self.bic_patience = bic_patience
        self.min_stages = min_stages
        self.freeze_decomposition = freeze_decomposition
        self.uniform_weights = uniform_weights
        self.global_tree_mode = global_tree_mode
        self.block1_mode = block1_mode
        self.random_state = random_state

    def _make_loss(self, y: np.ndarray) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _is_classification(self) -> bool:
        return False

    def _make_oof_folds(self, y: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
        if not self.use_oof or N < 4 or self.oof_folds < 2:
            return np.zeros(N, dtype=np.int64)
        n_splits = min(self.oof_folds, N)
        folds = np.zeros(N, dtype=np.int64)
        seed = int(rng.integers(0, 2**31 - 1))
        if self._is_classification():
            _, counts = np.unique(y, return_counts=True)
            min_count = int(np.min(counts))
            if min_count >= 2:
                n_splits_strat = min(n_splits, min_count)
                splitter = StratifiedKFold(n_splits=n_splits_strat, shuffle=True, random_state=seed)
                for q, (_, val_idx) in enumerate(splitter.split(np.zeros(N), y)):
                    folds[val_idx] = q
                return folds
        splitter_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for q, (_, val_idx) in enumerate(splitter_kf.split(np.zeros(N))):
            folds[val_idx] = q
        return folds

    def _fit_core(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> AdaSigBoostBase:
        if _adasigboost_cpp is None:
            raise RuntimeError("C++ extension _adasigboost_cpp is not built. Run `make build`.")

        X_3d, Z_arr = check_signal_and_tabular(X, Z)
        N, C, T = X_3d.shape
        self._input_ndim_ = 3 if (isinstance(X, np.ndarray) and X.ndim == 3) else 2
        if self._input_ndim_ == 2:
            validate_data(self, X, accept_sparse=False, dtype=np.float64, reset=True)
        self.n_channels_ = C
        self.signal_length_ = T
        self.n_tabular_features_ = 0 if Z_arr is None else Z_arr.shape[1]

        rng = set_deterministic_seed(self.random_state)
        loss_fn = self._make_loss(y)
        L_out = loss_fn.n_outputs

        self.init_pred_ = loss_fn.init_prediction(y)
        F = np.tile(self.init_pred_[np.newaxis, :], (N, 1))

        # Precompute signal FFTs and energies once (float32/complex64 for fast Block 1)
        X_fft_np = np.fft.rfft(X_3d, axis=-1)  # (N, C, F_freq)
        # Keep PyTorch CPU thread count bounded to avoid OpenMP contention with _adasigboost_cpp
        if torch.get_num_threads() > 2:
            torch.set_num_threads(2)
        X_fft_torch = torch.tensor(X_fft_np, dtype=torch.complex64)
        X_power_np = np.mean(X_fft_np.real**2 + X_fft_np.imag**2, axis=1)  # (N, F_freq)
        raw_energy = np.mean(X_3d * X_3d, axis=(1, 2))  # (N,)

        if Z_arr is not None:
            Z_binned, Z_edges = _adasigboost_cpp.compute_quantile_bins(Z_arr, self.max_bins)
        else:
            Z_binned, Z_edges = None, None

        folds = self._make_oof_folds(y, N, rng)
        eff_min_leaf = min(self.min_samples_leaf, max(1, N // 4))

        self.stages_: list[StageModel] = []
        self.stage_losses_: list[float] = [loss_fn.loss(y, F)]
        self.mode_weight_history_: list[np.ndarray] = []
        self.active_mode_counts_: list[int] = []

        prev_params: FilterBankParams | None = None
        prev_active_mask: np.ndarray | None = None
        bic_stall_count = 0

        for m in range(self.n_estimators):
            g, h = loss_fn.grad_hess(y, F)  # (N, L_out)

            # Block 1: OOF Residual-Guided Adaptive Signal Breakdown
            filter_params, s_ch, s_mix = fit_stage_filterbank(
                X_fft_np=X_fft_np,
                X_fft_torch=X_fft_torch,
                X_power_np=X_power_np,
                g_np=g,
                h_np=h,
                folds=folds,
                n_modes=self.n_modes,
                n_channels=C,
                signal_length=T,
                dictionary_type=self.dictionary_type,
                prev_params=prev_params,
                active_mask=prev_active_mask,
                alpha1=self.alpha1,
                alpha2=self.alpha2,
                n_steps=self.n_filter_steps,
                lr=self.filter_lr,
                use_oof=self.use_oof,
                freeze_init=self.freeze_decomposition,
                block1_mode=self.block1_mode,
            )

            # Step 3: Extract physical descriptors Φ(s_{i,m,k}) -> (N, K, d_Φ)
            desc = extract_descriptors(s_mix=s_mix, s_ch=s_ch, raw_energy=raw_energy)

            if self.global_tree_mode:
                # ABL-3: Pool all modes into one shared global tree per output
                X_pool = desc.reshape(N, -1)
                pool_binned, pool_edges = _adasigboost_cpp.compute_quantile_bins(
                    X_pool, self.max_bins
                )
                global_trees: list[Any | None] = []
                delta_F = np.zeros((N, L_out), dtype=np.float64)
                for c in range(L_out):
                    tree = _adasigboost_cpp.fit_hist_tree(
                        pool_binned,
                        pool_edges,
                        g[:, c],
                        h[:, c],
                        self.max_depth,
                        self.reg_lambda,
                        self.reg_gamma,
                        self.min_child_hessian,
                        eff_min_leaf,
                    )
                    global_trees.append(tree)
                    delta_F[:, c] = tree.predict(X_pool)

                tab_trees = None
                if Z_arr is not None and Z_binned is not None and Z_edges is not None:
                    tab_trees = []
                    for c in range(L_out):
                        g_tab = g[:, c] + h[:, c] * delta_F[:, c]
                        t_tab = _adasigboost_cpp.fit_hist_tree(
                            Z_binned,
                            Z_edges,
                            g_tab,
                            h[:, c],
                            self.max_depth,
                            self.reg_lambda,
                            self.reg_gamma,
                            self.min_child_hessian,
                            eff_min_leaf,
                        )
                        tab_trees.append(t_tab)
                        delta_F[:, c] += t_tab.predict(Z_arr)

                w_mat = np.full((self.n_modes, L_out), 1.0 / self.n_modes, dtype=np.float64)
                stage = StageModel(
                    filter_params=filter_params,
                    mode_trees=[global_trees],
                    tabular_trees=tab_trees,
                    weights=w_mat,
                    active_modes=list(range(self.n_modes)),
                    global_tree_mode=True,
                )
                F = F + self.learning_rate * delta_F
                self.stages_.append(stage)
                self.stage_losses_.append(loss_fn.loss(y, F))
                self.mode_weight_history_.append(w_mat.copy())
                self.active_mode_counts_.append(self.n_modes)
                prev_params = filter_params
                prev_active_mask = np.ones(self.n_modes, dtype=bool)
                continue

            # Pre-bin each mode's descriptor matrix once for Block 2 inner iterations
            mode_binned = []
            for k in range(self.n_modes):
                xb, xe = _adasigboost_cpp.compute_quantile_bins(desc[:, k, :], self.max_bins)
                mode_binned.append((xb, xe))

            # Frequency-band dropout mask
            keep_mask: np.ndarray = np.ones(self.n_modes, dtype=bool)
            if self.mode_dropout > 0.0 and self.n_modes > 1:
                keep_mask = np.asarray(rng.random(self.n_modes) >= self.mode_dropout, dtype=bool)
                if not np.any(keep_mask):
                    keep_mask[int(rng.integers(0, self.n_modes))] = True
            drop_scale = 1.0 / max(1.0 - self.mode_dropout, 0.1) if self.mode_dropout > 0.0 else 1.0

            w_mat = np.zeros((self.n_modes, L_out), dtype=np.float64)
            w_mat[keep_mask, :] = (1.0 / np.sum(keep_mask)) * drop_scale

            mode_trees: list[list[Any | None]] = [
                [None for _ in range(L_out)] for _ in range(self.n_modes)
            ]
            Z_preds = np.zeros((N, self.n_modes, L_out), dtype=np.float64)
            stage_tab_trees: list[Any] | None = None
            tab_preds = np.zeros((N, L_out), dtype=np.float64)

            n_inner = 1 if self.uniform_weights else max(1, self.n_inner_iters)
            for t_in in range(n_inner):
                # Block 2: Exact coordinate-adjusted tabular + per-signal trees
                for c in range(L_out):
                    # Maintain total current increment for output c
                    delta_c = np.sum(Z_preds[:, :, c] * w_mat[:, c][np.newaxis, :], axis=1) + tab_preds[:, c]
                    # Fit tabular branch v_{m,0}(z_i) first so static covariates explain baseline state
                    if Z_arr is not None and Z_binned is not None and Z_edges is not None:
                        if stage_tab_trees is None:
                            stage_tab_trees = [None] * L_out
                        delta_excl_tab = delta_c - tab_preds[:, c]
                        g_tab = g[:, c] + h[:, c] * delta_excl_tab
                        cand_tab = _adasigboost_cpp.fit_hist_tree(
                            Z_binned,
                            Z_edges,
                            g_tab,
                            h[:, c],
                            self.max_depth,
                            self.reg_lambda,
                            self.reg_gamma,
                            self.min_child_hessian,
                            eff_min_leaf,
                        )
                        cand_tab_pred = cand_tab.predict(Z_arr)
                        old_tab = stage_tab_trees[c]
                        if old_tab is None or (
                            _evaluate_tree_subobj(g_tab, h[:, c], cand_tab_pred, cand_tab)
                            <= _evaluate_tree_subobj(g_tab, h[:, c], tab_preds[:, c], old_tab) + 1e-12
                        ):
                            stage_tab_trees[c] = cand_tab
                            delta_c = delta_excl_tab + cand_tab_pred
                            tab_preds[:, c] = cand_tab_pred

                    for k in range(self.n_modes):
                        if not keep_mask[k]:
                            continue
                        wk = w_mat[k, c]
                        if abs(wk) < 1e-12:
                            continue
                        delta_excl = delta_c - wk * Z_preds[:, k, c]
                        g_eff = wk * (g[:, c] + h[:, c] * delta_excl)
                        h_eff = h[:, c] * (wk * wk)

                        xb, xe = mode_binned[k]
                        cand_tree = _adasigboost_cpp.fit_hist_tree(
                            xb,
                            xe,
                            g_eff,
                            h_eff,
                            self.max_depth,
                            self.reg_lambda,
                            self.reg_gamma,
                            self.min_child_hessian,
                            eff_min_leaf,
                        )
                        cand_pred = cand_tree.predict(desc[:, k, :])

                        old_tree = mode_trees[k][c]
                        if old_tree is None or (
                            _evaluate_tree_subobj(g_eff, h_eff, cand_pred, cand_tree)
                            <= _evaluate_tree_subobj(g_eff, h_eff, Z_preds[:, k, c], old_tree) + 1e-12
                        ):
                            mode_trees[k][c] = cand_tree
                            delta_c = delta_excl + wk * cand_pred
                            Z_preds[:, k, c] = cand_pred

                # Block 3: Closed-form second-order Elastic-Net / Group-Elastic-Net weights
                if not self.uniform_weights:
                    Z_masked = Z_preds.copy()
                    Z_masked[:, ~keep_mask, :] = 0.0
                    g_for_w = g + h * tab_preds
                    if L_out == 1:
                        w_opt, _, _ = _adasigboost_cpp.solve_elastic_net_weights(
                            Z_masked[:, :, 0],
                            g_for_w[:, 0],
                            h[:, 0],
                            w_mat[:, 0],
                            self.lambda1,
                            self.lambda2,
                            500,
                            1e-7,
                        )
                        w_mat = w_opt[:, np.newaxis]
                    else:
                        w_mat, _, _ = _adasigboost_cpp.solve_group_elastic_net_weights(
                            Z_masked,
                            g_for_w,
                            h,
                            w_mat,
                            self.lambda1,
                            self.lambda2,
                            500,
                            1e-7,
                        )
                    w_mat[~keep_mask, :] = 0.0
                    if self.mode_dropout > 0.0 and t_in == n_inner - 1:
                        w_mat = w_mat * drop_scale

            # Active Mode Pruning
            active_modes = [k for k in range(self.n_modes) if float(np.linalg.norm(w_mat[k])) > 1e-12]
            if not active_modes:
                break

            # Discard trees for pruned modes
            active_set = set(active_modes)
            for k in range(self.n_modes):
                if k not in active_set:
                    mode_trees[k] = [None] * L_out
                    w_mat[k, :] = 0.0

            delta_F = np.einsum("nkc,kc->nc", Z_preds, w_mat) + tab_preds
            F_next = F + self.learning_rate * delta_F
            loss_prev = self.stage_losses_[-1]
            loss_next = loss_fn.loss(y, F_next)

            stage = StageModel(
                filter_params=filter_params,
                mode_trees=mode_trees,
                tabular_trees=stage_tab_trees,
                weights=w_mat.copy(),
                active_modes=active_modes,
                global_tree_mode=False,
            )
            F = F_next
            self.stages_.append(stage)
            self.stage_losses_.append(loss_next)
            self.mode_weight_history_.append(w_mat.copy())
            self.active_mode_counts_.append(len(active_modes))

            prev_params = filter_params
            prev_active_mask = np.array([k in active_set for k in range(self.n_modes)], dtype=bool)

            # BIC / MDL Early Stopping
            if self.use_bic_stopping and (m + 1) >= self.min_stages:
                gain_nll = float(N) * (loss_prev - loss_next)
                mdl_thresh = (
                    self.bic_scale
                    * 0.5
                    * len(active_modes)
                    * L_out
                    * ((2**self.max_depth) + 3)
                    * np.log(max(float(N), 2.0))
                )
                if gain_nll < mdl_thresh:
                    bic_stall_count += 1
                    if bic_stall_count >= self.bic_patience:
                        break
                else:
                    bic_stall_count = 0

        self.n_stages_fitted_ = len(self.stages_)
        # Expose sklearn-compatible feature_importances_ (mean absolute mode weight)
        if self.mode_weight_history_:
            mean_abs = np.mean(
                [np.linalg.norm(w, axis=1) for w in self.mode_weight_history_], axis=0
            )
            tot = float(np.sum(mean_abs))
            self.mode_importances_ = mean_abs / tot if tot > 0 else mean_abs
        else:
            self.mode_importances_ = np.zeros(self.n_modes, dtype=np.float64)

        return self

    def _raw_predict_and_contributions(
        self,
        X: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute raw margin F_M(X, Z) and per-mode contributions (N, K, L_out)."""
        if getattr(self, "_input_ndim_", 2) == 2 and not (isinstance(X, np.ndarray) and X.ndim == 3):
            validate_data(self, X, accept_sparse=False, dtype=np.float64, reset=False)
        X_3d, Z_arr = check_signal_and_tabular(X, Z)
        N, C, T = X_3d.shape
        if C != self.n_channels_ or T != self.signal_length_:
            raise ValueError(
                f"Expected signal shape (*, {self.n_channels_}, {self.signal_length_}), "
                f"got (*, {C}, {T})."
            )
        if self.n_tabular_features_ > 0 and Z_arr is None:
            raise ValueError("Model was fitted with tabular covariates Z; Z must be provided.")
        if Z_arr is not None and Z_arr.shape[1] != self.n_tabular_features_:
            raise ValueError(
                f"Expected Z with {self.n_tabular_features_} features, got {Z_arr.shape[1]}."
            )

        L_out = len(self.init_pred_)
        F = np.tile(self.init_pred_[np.newaxis, :], (N, 1))
        mode_contribs = np.zeros((N, self.n_modes, L_out), dtype=np.float64)

        if not self.stages_:
            return F, mode_contribs

        X_fft_np = np.fft.rfft(X_3d, axis=-1)
        raw_energy = np.mean(X_3d * X_3d, axis=(1, 2))

        for stage in self.stages_:
            s_ch, s_mix = apply_filterbank_np(X_fft_np, stage.filter_params)
            desc = extract_descriptors(s_mix=s_mix, s_ch=s_ch, raw_energy=raw_energy)

            if stage.global_tree_mode:
                X_pool = desc.reshape(N, -1)
                for c in range(L_out):
                    t_c = stage.mode_trees[0][c]
                    if t_c is not None:
                        pred_c = self.learning_rate * t_c.predict(X_pool)
                        F[:, c] += pred_c
                        mode_contribs[:, :, c] += pred_c[:, np.newaxis] / self.n_modes
            else:
                for k in stage.active_modes:
                    for c in range(L_out):
                        t_kc = stage.mode_trees[k][c]
                        if t_kc is None:
                            continue
                        wk = stage.weights[k, c]
                        if abs(wk) < 1e-12:
                            continue
                        contrib = self.learning_rate * wk * t_kc.predict(desc[:, k, :])
                        F[:, c] += contrib
                        mode_contribs[:, k, c] += contrib

            if stage.tabular_trees is not None and Z_arr is not None:
                for c in range(L_out):
                    F[:, c] += self.learning_rate * stage.tabular_trees[c].predict(Z_arr)

        return F, mode_contribs

    def predict_mode_contributions(
        self,
        X: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return per-sample, per-mode cumulative additive contributions (N, K, L_out)."""
        _, contribs = self._raw_predict_and_contributions(X, Z)
        return contribs
