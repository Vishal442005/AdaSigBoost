"""One-command end-to-end reproducibility pipeline (benchmarks -> case studies -> tables -> paper PDF)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmarks.case_studies import (
    run_cwru_case_study,
    run_ecg_case_study,
    run_low_data_curve,
    run_multimodal_ablation,
)
from benchmarks.ood_stress import run_ood_stress_suite
from benchmarks.run_ablations import run_ablation_suite
from benchmarks.run_benchmark import run_benchmark
from benchmarks.run_synthetic_suite import run_synthetic_suite
from paper.scripts.generate_tables import main as generate_tables


def main() -> None:
    print("=== Step 1: Running Phase 4 Controlled Synthetic Suite ===", flush=True)
    run_synthetic_suite(
        n_train=600,
        n_test=300,
        signal_length=512,
        snr_levels=[20.0, 10.0, 5.0, 0.0, -5.0],
        seeds=[0, 1],
    )

    print("=== Step 2: Running Phase 6 Systematic Ablations (ABL-1..ABL-7) ===", flush=True)
    run_ablation_suite(
        n_train=600,
        n_test=300,
        signal_length=512,
        seeds=[0, 1],
    )

    print("=== Step 3: Running Phase 5 Real-World Benchmarks ===", flush=True)
    run_benchmark(
        datasets=[
            "ECG5000",
            "eeg_eyestate",
            "Epilepsy",
            "cwru_bearing",
            "Wafer",
            "etth1_energy",
            "FordA",
            "InsectWingbeatSound",
        ],
        models=[
            "AdaSigBoost",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "DWT+LightGBM",
            "EMD+XGBoost",
            "VMD+CatBoost",
            "MultiRocket+Ridge",
            "MultiRocket+LightGBM",
            "LearnableFilter+MLP",
            "LearnableFilter+LightGBM",
            "PatchTST",
            "ModernTCN",
            "TimesNet",
            "iTransformer",
        ],
        n_folds=3,
        n_optuna_trials=0,
        max_samples=750,
    )

    print("=== Step 4: Running Case Studies & OOD Stress Tests ===", flush=True)
    run_cwru_case_study()
    run_ecg_case_study()
    run_multimodal_ablation()
    run_low_data_curve()
    run_ood_stress_suite()

    print("=== Step 5: Auto-Generating Paper LaTeX Tables ===", flush=True)
    generate_tables()

    print("=== Step 6: Compiling Paper PDF ===", flush=True)
    paper_dir = Path("paper")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir, check=True)
    subprocess.run(["bibtex", "main"], cwd=paper_dir, check=True)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
    print("Reproducibility pipeline complete! Output: paper/main.pdf", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
