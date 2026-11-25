# AdaSigBoost: 10/10 Perfection Master Plan

This document upgrades the baseline research plan into a Tier-1 Oral/Spotlight-caliber execution roadmap targeting **ICML / NeurIPS / IEEE TPAMI**, elevating every review dimension to 10/10.

---

## Current Workspace Status (Completed So Far)
- **Phase 0 Audit:** Literature novelty verified; 5 mathematical/algorithmic bugs in the `.docx` resolved; hardware & dataset constraints mapped.
- **Phase 1 Theory Note (v1):** [docs/theory/AdaSigBoost_Theory_Note.tex](docs/theory/AdaSigBoost_Theory_Note.tex) compiled cleanly to PDF with Lemma 1, Theorem 1, and Theorem 2. *(Will be upgraded in v2 below with Theorem 3 and multi-channel/group-sparse extensions.)*
- **Phase 2 Scaffolding & Build System:** Git/CI/packaging configured (`pyproject.toml`, `CMakeLists.txt`, `Makefile`, `.github/workflows/ci.yml`); virtualenv `/home/jarvis/.venvs/adasigboost` created with `xgboost`, `lightgbm`, `optuna`, `statsmodels`, `vmdpy`, `EMD-signal`; C++17/OpenMP backend `_adasigboost_cpp` built and verified.

---

## 10/10 Perfection Upgrades Across All Dimensions

### 1. Novelty & Architectural Depth (8.5 -> 10/10)
1. **Composite Multi-Physics Adaptive Dictionary ($\Psi$):**
   - *Oscillatory / Chirp Atoms:* Unit-$L^2$-normalized Gabor-Morlet FFT filterbanks $(\omega_{m,k}, \sigma_{m,k})$.
   - *Trend / Baseline Atoms:* Learnable smooth low-pass / polynomial-difference atoms so slow secular trends and baseline wander separate cleanly without Gabor boundary ringing.
   - *Rational / Wavelet-Packet Atoms:* Selectable sharp-cutoff spectral atoms for narrowband harmonic faults.
2. **Spatial-Spectral Multi-Channel Mixing ($C > 1$):**
   - Learnable channel-attention weights $a_{m,k} \in \Delta^{C-1}$ (simplex-normalized via softmax) plus cross-channel coherence descriptors in $\Phi$, enabling lead-aware ECG and multi-electrode EEG decomposition.
3. **True Joint Multi-Class Boosting with Group-Sparse Modes:**
   - Support joint multi-class softmax boosting with shared physical filterbank $\Theta_m$ and class-coupled mode weight matrix $W_m \in \mathbb{R}^{K \times C_{\text{cls}}}$ regularized by group-Lasso ($\lambda_1 \sum_{k=1}^K \|W_{m,k,:}\|_2 + \frac{\lambda_2}{2}\|W_m\|_F^2$), selecting a sparse set of shared physical modes across classes.

### 2. Algorithmic Soundness & Systems Speed (8.5 -> 10/10)
4. **Residual Cross-Spectral Peak Initialization + Exact Block 1 Objective:**
   - Initialize pruned or fresh filter atoms directly at top peaks of the $h$-weighted cross-spectral residual density $\left|\sum_i g_{i,m} |\hat{x}_i(\omega)|^2\right|$ before short L-BFGS/Adam refinement (10–15 steps).
   - Maximize $h$-weighted correlation of differentiable summary statistics $\tilde{\Phi}(s_{i,m,k})$ (soft-max Hilbert envelope, local energy variance, spectral flux, kurtosis) against OOF Newton pseudo-residuals $r_{i,m}^{(-q)}$.
5. **Exact Block 2 Cross-Term + Faster C++17/OpenMP Backend:**
   - Block 2 trees use coordinate-adjusted $\tilde{g}_{i,m,k}$ and $\tilde{h}_{i,m,k}$ (guaranteeing exact joint descent with Block 3).
   - Add **smaller-child histogram subtraction** in `hist_tree.cpp`, OpenMP parallelization across $K$ modes, and fast vectorized/C++ descriptor extraction so inference latency rivals `LightGBM`.

### 3. Theoretical Rigor (8.5 -> 10/10)
6. **Three Complete Theorems + Support Recovery Proposition (`AdaSigBoost_Theory_Note.tex` v2):**
   - **Lemma 1:** Exact monotonic non-increase of $J_m$ across Block 2 & Block 3.
   - **Theorem 1:** Global geometric convergence on bounded iterates $[-B, B]$ tracking the explicit regularization floor $(\lambda_1, \lambda_2, \gamma, \beta)$.
   - **Proposition 1 (Mode Support Recovery):** Under bounded mutual incoherence between signal modes and uncorrelated distractors, soft-thresholding $\lambda_1$ recovers the true active mode set with high probability while zeroing out distractors.
   - **Theorem 2 (Multi-Channel & Group-Sparse Rademacher Bound):** Extended covering-number + VC-dimension + convex-hull proof covering $C$-channel spatial mixing and group-sparse $W_m$, showing $K$ enters only logarithmically.
   - **Theorem 3 (OOF Residual Variance-Bias Reduction):** Formal proof quantifying how $Q$-fold cross-fitted residual decomposition eliminates $O(N^{-1/2})$ in-sample filter selection overfitting bias.

### 4. Empirical & Experimental Rigor (9.0 -> 10/10)
7. **14-Baseline Competitive Matrix (Identical 50-Trial Optuna Budget):**
   - *Family A (Raw GBDTs):* XGBoost, LightGBM, CatBoost.
   - *Family B (Static 2-Stage + GBDT):* DWT(db4)+LightGBM, CEEMDAN+XGBoost, VMD+CatBoost.
   - *Family C (Learnable Neural Filterbanks — Direct Hybrid Competitors):* Learnable Group Transform/SincNet + MLP, Learnable Group Transform + LightGBM.
   - *Family D (Convolutional Kernels):* MultiRocket+Ridge, MultiRocket+LightGBM.
   - *Family E (Deep Time-Series SOTA):* TimesNet, PatchTST, iTransformer, ModernTCN (+ appendix linear-probe comparison vs. `MOMENT`/`Chronos` where feasible).
8. **Leak-Free Multi-Domain Benchmarks + Multimodal & Low-Data Regimes:**
   - 8 datasets across Cardiology (PTB-XL, MIT-BIH), Neurology (CHB-MIT / Sleep-EDF + optional TUSZ), Industrial Prognostics (CWRU, NASA IMS), Energy+Tabular (UCI Electricity + Wind SCADA + weather covariates), Finance (FI-2010 LOB), and Acoustics (ESC-50, UrbanSound8K) — all using strict subject-disjoint, load-disjoint, or temporal splits.
   - **Multimodal Signal + Tabular Ablation:** Explicitly prove `v_{m,0}(z_i)` + signal modes beats pure time-series SOTA on PTB-XL+demographics and Wind+weather.
   - **Low-Data Learning Curves:** Evaluate $N \in \{100, 500, 1000, 5000, \text{full}\}$ to showcase sample efficiency over transformers.
   - **Out-of-Distribution Noise Stress Tests:** Colored $1/f^\beta$ noise and sensor drift injection at test time.

### 5. Quantitative Interpretability & Statistical Proof (9.0 -> 10/10)
9. **Measurable Interpretability (Not Just Qualitative Plots):**
   - **Mode Deletion / Insertion Faithfulness AUC** (proving $w_{m,k}$ ranks truly causal signal modes).
   - **Physical Fault Frequency Error ($\Delta f$ in Hz)** comparing learned $\omega_{m,k}$ against analytical CWRU BPFO/BPFI harmonics.
   - **Friedman + Nemenyi CD Diagram + Holm-Bonferroni Wilcoxon + ECE/Brier Calibration + Accuracy-vs-Latency Pareto Frontier.**

### 6. Software Engineering, Manuscript & GitHub Release (9.0 -> 10/10)
10. **End-to-End Automated Pipeline:**
    - 100% `sklearn.utils.estimator_checks.check_estimator` pass + >90% test coverage.
    - `paper/scripts/generate_tables.py` auto-generates all LaTeX tables and figures directly from logged JSON/CSV results (zero manual copy-paste).
    - Complete NeurIPS/ICML 9-page main paper + Appendices A–E + NeurIPS Reproducibility Checklist + Red-Team Q&A audit.
    - Clean MIT-licensed GitHub repository with CI (`ruff`, `mypy`, `pytest`, C++ build), docs, quickstart notebook, and tagged `v0.1.0-paper` release.

---

## Execution Order from Here
1. **Upgrade Phase 1 (`AdaSigBoost_Theory_Note.tex` v2):** Add Theorem 3 (OOF bias reduction), Proposition 1 (support recovery), multi-channel spatial mixing, and group-sparse multi-class extension.
2. **Complete Phase 3 (Core Engine):** `decomposition/` (Gabor-Morlet + Trend + Rational + Channel Attention) -> `descriptors/` -> `csrc/` (histogram subtraction + OpenMP mode loop) -> `core/` (Binary, Multi-class Group-Sparse, Regressor, OOF engine, BIC stopper, `check_estimator` pass) -> `interpretability/`.
3. **Execute Phase 4 (Synthetic Suite & Theory Gates):** Run KKT checks, Lemma 1 monotonic descent test, $>0.95$ IoU mode recovery, distractor zeroing, SNR sweeps ($+20$ to $-5\text{ dB}$), and $K$-sensitivity.
4. **Execute Phase 5 & 6 (Real Benchmarks, 14 Baselines, 7 Ablations, Quantitative Interpretability):** Run local CPU benchmarks + export GPU-ready cloud scripts for Family E transformers; compute Friedman/Nemenyi CD diagrams, Pareto plots, deletion AUC, and CWRU/PTB-XL physical attributions.
5. **Execute Phase 7 & 8 (Manuscript + GitHub Push):** Generate figures/tables, write 9-page LaTeX paper + Appendices A–E, red-team audit, polish repo/CI, and push to GitHub.
