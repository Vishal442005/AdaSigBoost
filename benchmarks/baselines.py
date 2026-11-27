"""Complete baseline model wrappers across Families A-E (12 baselines)."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pywt
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler


def _extract_raw_summary_stats(X: np.ndarray) -> np.ndarray:
    """Flatten raw signals and append basic time/spectral summary statistics."""
    X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
    N = X_3d.shape[0]
    flat = X_3d.reshape(N, -1)
    means = np.mean(X_3d, axis=-1)
    stds = np.std(X_3d, axis=-1)
    mins = np.min(X_3d, axis=-1)
    maxs = np.max(X_3d, axis=-1)
    return np.hstack([flat, means, stds, mins, maxs])


class RawGBDTBaseline:
    """Family A: Raw GBDT (XGBoost, LightGBM, CatBoost) on flattened signals + summary stats."""

    def __init__(self, backend: str = "xgboost", task: str = "regression", **params: Any) -> None:
        self.backend = backend
        self.task = task
        self.params = params
        self.model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> RawGBDTBaseline:
        feats = _extract_raw_summary_stats(X)
        if Z is not None:
            feats = np.hstack([feats, Z])
        if self.backend == "xgboost":
            cls = xgb.XGBRegressor if self.task == "regression" else xgb.XGBClassifier
            self.model = cls(n_jobs=4, verbosity=0, **self.params)
        elif self.backend == "lightgbm":
            cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
            self.model = cls(n_jobs=4, verbose=-1, **self.params)
        elif self.backend == "catboost":
            cls = CatBoostRegressor if self.task == "regression" else CatBoostClassifier
            self.model = cls(thread_count=4, verbose=0, allow_writing_files=False, **self.params)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        self.model.fit(feats, y)
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        feats = _extract_raw_summary_stats(X)
        if Z is not None:
            feats = np.hstack([feats, Z])
        pred = self.model.predict(feats)
        return np.asarray(pred).ravel()

    def predict_proba(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        feats = _extract_raw_summary_stats(X)
        if Z is not None:
            feats = np.hstack([feats, Z])
        return self.model.predict_proba(feats)


def _decompose_dwt_features(X: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Extract static DWT subband energy, std, and max features."""
    X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
    N, C, _ = X_3d.shape
    rows = []
    for i in range(N):
        feats_i = []
        for c in range(C):
            coeffs = pywt.wavedec(X_3d[i, c], wavelet=wavelet, level=level)
            for cf in coeffs:
                e = float(np.mean(cf**2))
                s = float(np.std(cf))
                m = float(np.max(np.abs(cf)))
                feats_i.extend([np.log(e + 1e-8), s, m])
        rows.append(feats_i)
    return np.asarray(rows, dtype=np.float64)


def _decompose_emd_features(X: np.ndarray, max_imfs: int = 4) -> np.ndarray:
    """Extract static Empirical Mode Decomposition (EMD) intrinsic mode descriptors."""
    from PyEMD import EMD

    X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
    N, C, _ = X_3d.shape
    emd = EMD(MAX_ITERATION=40)
    rows = []
    for i in range(N):
        feats_i = []
        for c in range(min(C, 4)):
            sig = X_3d[i, c]
            try:
                imfs = emd.emd(sig, max_imf=max_imfs)
            except Exception:
                imfs = np.zeros((1, len(sig)), dtype=np.float64)
            for k in range(max_imfs):
                if k < len(imfs):
                    uk = imfs[k]
                    e = float(np.mean(uk**2))
                    s = float(np.std(uk))
                    m = float(np.max(np.abs(uk)))
                    feats_i.extend([np.log(e + 1e-8), s, m])
                else:
                    feats_i.extend([-18.4, 0.0, 0.0])
        rows.append(feats_i)
    return np.asarray(rows, dtype=np.float64)


def _decompose_vmd_features(X: np.ndarray, K: int = 5, alpha: float = 2000.0) -> np.ndarray:
    """Extract static VMD intrinsic mode descriptors (energy, max envelope, center freq)."""
    from vmdpy import VMD

    X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
    N, C, _ = X_3d.shape
    rows = []
    for i in range(N):
        feats_i = []
        for c in range(min(C, 4)):
            sig = X_3d[i, c]
            try:
                u, _, omega = VMD(sig, alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=1e-3)
                for k in range(K):
                    uk = u[k]
                    e = float(np.mean(uk**2))
                    m = float(np.max(np.abs(uk)))
                    om = float(omega[-1, k]) if omega.ndim == 2 else float(omega[k])
                    feats_i.extend([np.log(e + 1e-8), m, om])
            except Exception:
                feats_i.extend([0.0] * (3 * K))
        rows.append(feats_i)
    return np.asarray(rows, dtype=np.float64)


class StaticDecompGBDTBaseline:
    """Family B: Open-loop static DWT/EMD/VMD decomposition followed by LightGBM/XGBoost/CatBoost."""

    def __init__(
        self,
        decomp: str = "dwt",
        gbdt: str = "lightgbm",
        task: str = "regression",
        n_modes: int = 5,
        **params: Any,
    ) -> None:
        self.decomp = decomp
        self.gbdt = gbdt
        self.task = task
        self.n_modes = n_modes
        self.params = params
        self.model: Any = None

    def _featurize(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        if self.decomp == "dwt":
            f = _decompose_dwt_features(X, level=max(2, self.n_modes - 1))
        elif self.decomp == "emd":
            f = _decompose_emd_features(X, max_imfs=min(4, self.n_modes))
        elif self.decomp == "vmd":
            f = _decompose_vmd_features(X, K=self.n_modes)
        else:
            raise ValueError(f"Unknown static decomp: {self.decomp}")
        if Z is not None:
            f = np.hstack([f, Z])
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> StaticDecompGBDTBaseline:
        feats = self._featurize(X, Z)
        if self.gbdt == "lightgbm":
            cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
            self.model = cls(n_jobs=4, verbose=-1, **self.params)
        elif self.gbdt == "xgboost":
            cls = xgb.XGBRegressor if self.task == "regression" else xgb.XGBClassifier
            self.model = cls(n_jobs=4, verbosity=0, **self.params)
        elif self.gbdt == "catboost":
            cls = CatBoostRegressor if self.task == "regression" else CatBoostClassifier
            self.model = cls(thread_count=4, verbose=0, allow_writing_files=False, **self.params)
        else:
            raise ValueError(f"Unknown gbdt: {self.gbdt}")
        self.model.fit(feats, y)
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model.predict(self._featurize(X, Z))).ravel()

    def predict_proba(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return self.model.predict_proba(self._featurize(X, Z))


class RandomConvKernelBaseline:
    """Family D: ROCKET/MultiRocket-style multi-dilation random kernels + Ridge/LightGBM."""

    def __init__(
        self,
        n_kernels: int = 500,
        head: str = "ridge",
        task: str = "regression",
        seed: int = 42,
        **head_params: Any,
    ) -> None:
        self.n_kernels = n_kernels
        self.head = head
        self.task = task
        self.seed = seed
        self.head_params = head_params
        self.kernels_: list[tuple[np.ndarray, int, float]] = []
        self.scaler = StandardScaler()
        self.model: Any = None

    def _init_kernels(self, T: int) -> None:
        rng = np.random.default_rng(self.seed)
        lengths = [7, 9, 11]
        self.kernels_ = []
        max_dil = max(1, (T - 1) // 11)
        for _ in range(self.n_kernels):
            k_len = int(rng.choice(lengths))
            w = rng.normal(size=k_len)
            w = w - np.mean(w)
            dil = int(2 ** rng.uniform(0.0, np.log2(max_dil + 1)))
            bias = float(rng.uniform(-1.0, 1.0))
            self.kernels_.append((w, dil, bias))

    def _transform(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
        N, _, T = X_3d.shape
        x_mean = np.mean(X_3d, axis=1)
        feats = np.zeros((N, 2 * len(self.kernels_)), dtype=np.float64)
        for idx, (w, dil, bias) in enumerate(self.kernels_):
            k_len = len(w)
            span = (k_len - 1) * dil + 1
            if span > T:
                continue
            conv = np.zeros((N, T - span + 1), dtype=np.float64)
            for j in range(k_len):
                conv += w[j] * x_mean[:, j * dil : j * dil + (T - span + 1)]
            conv += bias
            feats[:, 2 * idx] = np.mean(conv > 0.0, axis=1)
            feats[:, 2 * idx + 1] = np.max(conv, axis=1)
        if Z is not None:
            feats = np.hstack([feats, Z])
        return feats

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> RandomConvKernelBaseline:
        T = X.shape[-1]
        self._init_kernels(T)
        feats = self.scaler.fit_transform(self._transform(X, Z))
        if self.head == "ridge":
            cls = Ridge if self.task == "regression" else RidgeClassifier
            self.model = cls(**self.head_params)
        else:
            cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
            self.model = cls(n_jobs=4, verbose=-1, **self.head_params)
        self.model.fit(feats, y)
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        feats = self.scaler.transform(self._transform(X, Z))
        return np.asarray(self.model.predict(feats)).ravel()

    def predict_proba(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        feats = self.scaler.transform(self._transform(X, Z))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(feats)
        scores = self.model.decision_function(feats)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))
            return np.column_stack([1.0 - p1, p1])
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)


class _SimpleLearnableFilterNet(nn.Module):
    """Differentiable Gabor filterbank + MLP proxy baseline."""

    def __init__(self, T: int, n_filters: int = 8, n_outputs: int = 1) -> None:
        super().__init__()
        self.T = T
        self.n_filters = n_filters
        n_freq = T // 2 + 1
        self.register_buffer("freqs", torch.linspace(0.0, np.pi, n_freq))
        self.omega_raw = nn.Parameter(torch.linspace(-2.0, 2.0, n_filters))
        self.sigma_raw = nn.Parameter(torch.zeros(n_filters))
        self.head = nn.Sequential(
            nn.Linear(2 * n_filters, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_mean = torch.mean(x, dim=1)
        xf = torch.fft.rfft(x_mean, dim=-1)
        om = 0.05 + 0.9 * np.pi * torch.sigmoid(self.omega_raw)
        sg = 0.05 + 0.5 * torch.sigmoid(self.sigma_raw)
        H = torch.exp(-0.5 * ((self.freqs[None, :] - om[:, None]) / sg[:, None]) ** 2)
        s = torch.fft.irfft(xf.unsqueeze(1) * H.unsqueeze(0), n=self.T, dim=-1)
        e = torch.log(torch.mean(s * s, dim=-1) + 1e-6)
        m = torch.max(torch.abs(s), dim=-1).values
        return torch.cat([e, m], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))


class LearnableFilterNeuralBaseline:
    """Family C: End-to-end differentiable Gabor filterbank + MLP or LightGBM hybrid transfer."""

    def __init__(
        self,
        n_filters: int = 8,
        epochs: int = 30,
        lr: float = 1e-2,
        head: str = "mlp",
        task: str = "regression",
        seed: int = 42,
    ) -> None:
        self.n_filters = n_filters
        self.epochs = epochs
        self.lr = lr
        self.head = head
        self.task = task
        self.seed = seed
        self.net: _SimpleLearnableFilterNet | None = None
        self.gbdt_head: Any = None
        self.n_classes_ = 1

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> LearnableFilterNeuralBaseline:
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=device)
        if self.task == "regression":
            n_out = 1
            y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
            loss_fn: nn.Module = nn.MSELoss()
            self.n_classes_ = 1
        else:
            classes = np.unique(y)
            self.n_classes_ = len(classes)
            if self.n_classes_ <= 2:
                n_out = 1
                y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                n_out = self.n_classes_
                y_t = torch.tensor(y, dtype=torch.long, device=device)
                loss_fn = nn.CrossEntropyLoss()

        self.net = _SimpleLearnableFilterNet(
            X_3d.shape[-1], n_filters=self.n_filters, n_outputs=n_out
        ).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.net(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()

        if self.head == "lightgbm":
            self.net.eval()
            with torch.no_grad():
                feats = self.net.extract_features(X_t).cpu().numpy()
            if Z is not None:
                feats = np.hstack([feats, Z])
            cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
            self.gbdt_head = cls(n_estimators=100, max_depth=4, n_jobs=4, verbose=-1, random_state=self.seed)
            self.gbdt_head.fit(feats, y)
        return self

    def _export_feats(self, X: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        device = next(self.net.parameters()).device
        X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
        with torch.no_grad():
            feats = self.net.extract_features(torch.tensor(X_3d, dtype=torch.float32, device=device)).cpu().numpy()
        if Z is not None:
            feats = np.hstack([feats, Z])
        return feats

    def predict_proba(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        if self.head == "lightgbm" and self.gbdt_head is not None:
            return self.gbdt_head.predict_proba(self._export_feats(X, Z))
        assert self.net is not None
        self.net.eval()
        device = next(self.net.parameters()).device
        X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
        with torch.no_grad():
            logits = self.net(torch.tensor(X_3d, dtype=torch.float32, device=device)).cpu()
            if self.n_classes_ <= 2:
                p1 = torch.sigmoid(logits.squeeze(-1)).numpy()
                return np.column_stack([1.0 - p1, p1])
            return torch.softmax(logits, dim=-1).numpy()

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        if self.head == "lightgbm" and self.gbdt_head is not None:
            return np.asarray(self.gbdt_head.predict(self._export_feats(X, Z))).ravel()
        assert self.net is not None
        self.net.eval()
        device = next(self.net.parameters()).device
        X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
        with torch.no_grad():
            out = self.net(torch.tensor(X_3d, dtype=torch.float32, device=device)).cpu()
        if self.task == "regression":
            return out.squeeze(-1).numpy()
        if self.n_classes_ <= 2:
            return (out.squeeze(-1).numpy() > 0.0).astype(int)
        return np.argmax(out.numpy(), axis=1)


class _PatchTSTBackbone(nn.Module):
    """Family E: Compact PatchTST-style channel-independent Patched Transformer Encoder."""

    def __init__(
        self,
        n_channels: int,
        signal_length: int,
        n_outputs: int,
        n_tabular: int = 0,
        patch_len: int = 16,
        d_model: int = 48,
    ) -> None:
        super().__init__()
        self.patch_len = min(patch_len, max(4, signal_length // 4))
        self.stride = max(2, self.patch_len // 2)
        n_patches = max(1, (signal_length - self.patch_len) // self.stride + 1)
        self.proj = nn.Linear(self.patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(n_channels * d_model + n_tabular, n_outputs)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        N, C, T = x.shape
        if T < self.patch_len:
            x = nn.functional.pad(x, (0, self.patch_len - T))
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (N, C, P, patch_len)
        P = patches.shape[2]
        tok = self.proj(patches.reshape(N * C, P, self.patch_len))
        tok = tok + self.pos_emb[:, :P, :]
        enc = self.encoder(tok).mean(dim=1).reshape(N, C * tok.shape[-1])
        if z is not None:
            enc = torch.cat([enc, z], dim=-1)
        return self.head(enc)


class _ModernTCNBackbone(nn.Module):
    """Family E: Multi-scale depthwise-separable Temporal ConvNet (ModernTCN-style)."""

    def __init__(self, n_channels: int, n_outputs: int, n_tabular: int = 0, d_model: int = 48) -> None:
        super().__init__()
        self.stem = nn.Conv1d(n_channels, d_model, kernel_size=5, padding=2)
        self.block1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.head = nn.Linear(2 * d_model + n_tabular, n_outputs)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stem(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        pooled = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=-1)
        if z is not None:
            pooled = torch.cat([pooled, z], dim=-1)
        return self.head(pooled)


class _TimesNetBackbone(nn.Module):
    """Family E: Compact TimesNet-style FFT period-folding + 2D Inception temporal block."""

    def __init__(self, n_channels: int, n_outputs: int, n_tabular: int = 0, d_model: int = 32) -> None:
        super().__init__()
        self.stem = nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1)
        self.conv2d_a = nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1)
        self.conv2d_b = nn.Conv2d(d_model, d_model, kernel_size=(5, 5), padding=2)
        self.act = nn.GELU()
        self.head = nn.Linear(2 * d_model + n_tabular, n_outputs)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stem(x)  # (N, D, T)
        N, D, T = h.shape
        # Fold 1D features along dominant empirical periods
        periods = [max(4, min(T // 2, 16)), max(4, min(T // 3, 24))]
        accum = h
        for p in periods:
            pad = (p - (T % p)) % p
            hp = nn.functional.pad(h, (0, pad)) if pad > 0 else h
            cols = hp.shape[-1] // p
            h2d = hp.view(N, D, cols, p)
            y2d = 0.5 * (self.act(self.conv2d_a(h2d)) + self.act(self.conv2d_b(h2d)))
            accum = accum + y2d.reshape(N, D, cols * p)[:, :, :T]
        pooled = torch.cat([accum.mean(dim=-1), accum.max(dim=-1).values], dim=-1)
        if z is not None:
            pooled = torch.cat([pooled, z], dim=-1)
        return self.head(pooled)


class _iTransformerBackbone(nn.Module):
    """Family E: Compact iTransformer inverted variate-token attention encoder."""

    def __init__(self, n_channels: int, signal_length: int, n_outputs: int, n_tabular: int = 0, d_model: int = 48) -> None:
        super().__init__()
        self.variate_proj = nn.Linear(signal_length, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=96, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(n_channels * d_model + n_tabular, n_outputs)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        N, C, _ = x.shape
        tok = self.variate_proj(x)  # (N, C, d_model)
        enc = self.encoder(tok).reshape(N, C * tok.shape[-1])
        if z is not None:
            enc = torch.cat([enc, z], dim=-1)
        return self.head(enc)


class DeepTimeSeriesBaseline:
    """Family E: PyTorch PatchTST, ModernTCN, TimesNet, and iTransformer baselines (CUDA-aware)."""

    def __init__(
        self,
        arch: str = "patchtst",
        epochs: int = 25,
        lr: float = 3e-3,
        batch_size: int = 64,
        task: str = "regression",
        seed: int = 42,
    ) -> None:
        self.arch = arch
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.task = task
        self.seed = seed
        self.net: nn.Module | None = None
        self.scaler_x = StandardScaler()
        self.scaler_z = StandardScaler()
        self.n_classes_ = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prep_inputs(self, X: np.ndarray, Z: np.ndarray | None, fit: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
        N, C, T = X_3d.shape
        flat = X_3d.reshape(N, -1)
        scaled = self.scaler_x.fit_transform(flat) if fit else self.scaler_x.transform(flat)
        X_t = torch.tensor(scaled.reshape(N, C, T), dtype=torch.float32, device=self.device)
        Z_t = None
        if Z is not None:
            zs = self.scaler_z.fit_transform(Z) if fit else self.scaler_z.transform(Z)
            Z_t = torch.tensor(zs, dtype=torch.float32, device=self.device)
        return X_t, Z_t

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> DeepTimeSeriesBaseline:
        torch.manual_seed(self.seed)
        X_t, Z_t = self._prep_inputs(X, Z, fit=True)
        N, C, T = X_t.shape
        n_tab = 0 if Z_t is None else Z_t.shape[1]

        if self.task == "regression":
            n_out = 1
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(-1)
            loss_fn: nn.Module = nn.MSELoss()
            self.n_classes_ = 1
        else:
            self.n_classes_ = len(np.unique(y))
            if self.n_classes_ <= 2:
                n_out = 1
                y_t = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(-1)
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                n_out = self.n_classes_
                y_t = torch.tensor(y, dtype=torch.long, device=self.device)
                loss_fn = nn.CrossEntropyLoss()

        if self.arch == "patchtst":
            self.net = _PatchTSTBackbone(C, T, n_out, n_tabular=n_tab)
        elif self.arch == "moderntcn":
            self.net = _ModernTCNBackbone(C, n_out, n_tabular=n_tab)
        elif self.arch == "timesnet":
            self.net = _TimesNetBackbone(C, n_out, n_tabular=n_tab)
        elif self.arch == "itransformer":
            self.net = _iTransformerBackbone(C, T, n_out, n_tabular=n_tab)
        else:
            raise ValueError(f"Unknown deep arch: {self.arch}")
        self.net = self.net.to(self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.net.train()
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            perm = rng.permutation(N)
            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                zb = Z_t[idx] if Z_t is not None else None
                out = self.net(X_t[idx], zb)
                loss = loss_fn(out, y_t[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        X_t, Z_t = self._prep_inputs(X, Z, fit=False)
        with torch.no_grad():
            logits = self.net(X_t, Z_t).cpu()
            if self.n_classes_ <= 2:
                p1 = torch.sigmoid(logits.squeeze(-1)).numpy()
                return np.column_stack([1.0 - p1, p1])
            return torch.softmax(logits, dim=-1).numpy()

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        X_t, Z_t = self._prep_inputs(X, Z, fit=False)
        with torch.no_grad():
            out = self.net(X_t, Z_t).cpu()
        if self.task == "regression":
            return out.squeeze(-1).numpy()
        if self.n_classes_ <= 2:
            return (out.squeeze(-1).numpy() > 0.0).astype(int)
        return np.argmax(out.numpy(), axis=1)
