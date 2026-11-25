# Cloud GPU Execution Guide (Family E Deep Time-Series Baselines)

Local CPU hardware runs **AdaSigBoost**, **Family A** (XGBoost, LightGBM), **Family B** (DWT/EMD/VMD + GBDT), **Family C** (Learnable Filterbank + MLP), and **Family D** (MultiRocket + Ridge/LightGBM) directly.

For full 50-trial Optuna sweeps across **Family E** deep transformers (**TimesNet**, **PatchTST**, **iTransformer**, **ModernTCN**) on multi-domain benchmarks (PTB-XL, CWRU, ESC-50, ETTh1), use a single GPU (T4 / RTX 4090 / A100):

1. Clone this repository and install dependencies:
   ```bash
   pip install -e ".[benchmarks]"
   pip install neuralforecast tsai aeon
   ```
2. Export deterministic fold indices and run the shared evaluation harness so every baseline is scored on the exact same leak-free splits as `AdaSigBoost`.
3. Drop the resulting JSON logs into `results/` and run `python -m paper.scripts.generate_tables` to refresh all LaTeX tables automatically.
