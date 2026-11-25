"""One-command end-to-end reproducibility pipeline (benchmarks -> tables -> paper PDF)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
        datasets=["etth1_energy", "cwru_bearing", "ECG5000", "FordA"],
        models=[
            "AdaSigBoost",
            "XGBoost",
            "LightGBM",
            "DWT+LightGBM",
            "MultiRocket+Ridge",
            "LearnableFilter+MLP",
        ],
        n_folds=3,
        n_optuna_trials=5,
        max_samples=800,
    )

    print("=== Step 4: Auto-Generating Paper LaTeX Tables ===", flush=True)
    generate_tables()

    print("=== Step 5: Compiling Paper PDF ===", flush=True)
    paper_dir = Path("paper")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
    print("Reproducibility pipeline complete! Output: paper/main.pdf", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
