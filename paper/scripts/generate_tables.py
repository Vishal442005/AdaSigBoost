"""Auto-generate publication LaTeX tables directly from JSON benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path


def generate_synthetic_table(json_path: Path, out_tex: Path) -> None:
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text())["snr_summary"]
    snrs = ["20.0", "10.0", "5.0", "0.0", "-5.0"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Synthetic Ground-Truth Test RMSE ($\downarrow$) across Colored $1/f^\beta$ SNR Levels.}",
        r"\label{tab:synthetic_snr}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & $+20$ dB & $+10$ dB & $+5$ dB & $0$ dB & $-5$ dB \\",
        r"\midrule",
    ]
    for model_name, snr_dict in data.items():
        cells = []
        for s in snrs:
            if s in snr_dict:
                m = snr_dict[s]["rmse_mean"]
                sd = snr_dict[s]["rmse_std"]
                cells.append(f"${m:.3f} \\pm {sd:.3f}$")
            else:
                cells.append("--")
        row = f"{model_name} & " + " & ".join(cells) + r" \\"
        lines.append(row)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def generate_ablation_table(json_path: Path, out_tex: Path) -> None:
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text())
    snr20 = data.get("snr_20db", {})
    snr5 = data.get("snr_5db", {})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Systematic Ablation Study (ABL-1 through ABL-7) on Synthetic Signal Benchmarks.}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & $+20$ dB RMSE ($\downarrow$) & $+5$ dB RMSE ($\downarrow$) & Avg Active $K$ \\",
        r"\midrule",
    ]
    for v_name, m20 in snr20.items():
        m5 = snr5.get(v_name, {})
        r20 = f"${m20['rmse_mean']:.3f}$"
        r5 = f"${m5.get('rmse_mean', float('nan')):.3f}$"
        ak = f"${m5.get('avg_active_modes', m20['avg_active_modes']):.2f}$"
        v_tex = v_name.replace("λ1=0", r"$\lambda_1=0$").replace("α1=0", r"$\alpha_1=0$")
        lines.append(f"{v_tex} & {r20} & {r5} & {ak} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def generate_real_table(json_path: Path, out_tex: Path) -> None:
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text())
    ds_names = [k for k in data.keys() if not k.startswith("_")]
    if not ds_names:
        return
    model_names = list(data[ds_names[0]]["models"].keys())
    col_spec = "l" + "c" * len(ds_names)
    headers = " & ".join(d.replace("_", r"\_") for d in ds_names)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-Domain Real-World Benchmark Comparison (Mean $\pm$ Std across Leak-Free Folds).}",
        r"\label{tab:real_benchmark}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Method & {headers} \\\\",
        r"\midrule",
    ]
    for m in model_names:
        cells = []
        for d in ds_names:
            rec = data[d]["models"].get(m)
            if rec is None:
                cells.append("--")
            else:
                cells.append(f"${rec['primary_mean']:.3f} \\pm {rec['primary_std']:.3f}$")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def generate_stats_table(json_path: Path, out_tex: Path) -> None:
    if not json_path.exists():
        return
    raw = json.loads(json_path.read_text())
    stats = raw.get("_stats", {})
    wb = stats.get("wilcoxon_holm", [])
    fn = stats.get("friedman_nemenyi", {})
    if not wb:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Statistical Significance vs. AdaSigBoost (Friedman $p = {fn.get('p_value', 1.0):.4f}$, CD $= {fn.get('critical_difference_05', 0.0):.2f}$).}}",
        r"\label{tab:stats_sig}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Baseline & Mean Rank & Raw $p$ & Holm-Adjusted $p$ \\",
        r"\midrule",
    ]
    avg_ranks = fn.get("avg_ranks", {})
    for r in wb:
        b = r["baseline"]
        rk = avg_ranks.get(b, float("nan"))
        lines.append(f"{b} & ${rk:.2f}$ & ${r['p_raw']:.4f}$ & ${r['p_holm']:.4f}$ " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def generate_multimodal_table(json_path: Path, out_tex: Path) -> None:
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multimodal Signal + Tabular Covariate Fusion on \texttt{etth1\_energy} (24h Forecast MAE $\downarrow$).}",
        r"\label{tab:multimodal}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Input Modality & Test MAE ($\downarrow$) \\",
        r"\midrule",
    ]
    for name, val in data.items():
        lines.append(f"{name} & ${val:.4f}$ " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def main() -> None:
    results_dir = Path("results")
    tables_dir = Path("paper/tables")
    generate_synthetic_table(results_dir / "synthetic_suite.json", tables_dir / "tab_synthetic_snr.tex")
    generate_ablation_table(results_dir / "ablations.json", tables_dir / "tab_ablations.tex")
    generate_real_table(results_dir / "real_benchmark.json", tables_dir / "tab_real_benchmark.tex")
    generate_stats_table(results_dir / "real_benchmark.json", tables_dir / "tab_stats_significance.tex")
    generate_multimodal_table(results_dir / "multimodal_ablation.json", tables_dir / "tab_multimodal.tex")


if __name__ == "__main__":
    main()
