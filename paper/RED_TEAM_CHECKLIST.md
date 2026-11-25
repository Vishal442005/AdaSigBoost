# Pre-Submission Reviewer Red-Team Checklist (Phase 7)

This audit ensures all five anticipated Tier-1 reviewer critiques are preemptively answered with theorems, ablations, and empirical artifacts.

1. **Critique 1: "Isn't this just VMD + XGBoost?"**
   - *Answer:* Refuted by Section 1/3 (open-loop VMD optimizes signal reconstruction energy without label gradients, discarding low-energy transients), Theorem 1 + Proposition 1, and Ablation `ABL-1` (freezing decomposition drops accuracy sharply on low-energy transients and noisy mixtures).
2. **Critique 2: "Won't adaptive signal splitting overfit small datasets?"**
   - *Answer:* Refuted by Theorem 2 (logarithmic dependence on candidate mode count $K$ via $L_1$ budget $B_w$), Theorem 3 (5-fold OOF residual cross-fitting eliminates $O(N^{-1/2})$ selection bias), and Ablations `ABL-4` ($\lambda_1=0$) & `ABL-6` (in-sample residuals).
3. **Critique 3: "How sensitive is AdaSigBoost to choosing the number of modes $K$?"**
   - *Answer:* Included in `results/synthetic_suite.json` (`k_sensitivity_10db`) for $K \in \{2, 4, 8, 12, 16\}$, demonstrating that soft-thresholding prunes excess modes so performance plateaus smoothly without overfitting.
4. **Critique 4: "How does training & inference latency compare to LightGBM and neural baselines?"**
   - *Answer:* Logged per-model `train_time_s` and `infer_ms_per_sample` across all synthetic and real-world runs, enabling direct Accuracy-vs-Latency Pareto comparison.
5. **Critique 5: "Are learned sub-signals physically meaningful or arbitrary artifacts?"**
   - *Answer:* Quantified via Spectral IoU (`spectral_iou_20db`), Mode Deletion Faithfulness AUC (`compute_mode_deletion_auc`), and physical frequency alignment error ($\Delta f$ Hz against CWRU bearing harmonics and synthetic ground-truth modes).
