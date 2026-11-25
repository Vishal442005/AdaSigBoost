# Cloud GPU Execution Guide (Google Colab T4 / RTX 4090 / A100)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishal442005/AdaSigBoost/blob/main/notebooks/AdaSigBoost_GPU_Colab.ipynb)

Local CPU hardware runs **AdaSigBoost**, **Family A** (`XGBoost`, `LightGBM`, `CatBoost`), **Family B** (`DWT/EMD/VMD + GBDT`), **Family C** (`LearnableFilter + MLP/LightGBM`), **Family D** (`MultiRocket + Ridge/LightGBM`), and compact **Family E** (`PatchTST`, `ModernTCN`, `TimesNet`, `iTransformer`).

To run the full **5-fold × 50-trial Optuna sweep** (`--max-samples 3000`) with CUDA acceleration:
1. Click the **Open in Colab** badge above (`notebooks/AdaSigBoost_GPU_Colab.ipynb`) and select **Runtime → Change runtime type → T4 GPU**.
2. Execute all cells cleanly; Deep Time-Series baselines automatically utilize `cuda` when available.
3. Download `results_gpu.zip` and extract into `results/` and `paper/`, then run `python -m paper.scripts.generate_tables`.
