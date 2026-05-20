from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    ("continuation_regret", "Mean", "mean"),
    ("continuation_regret", "P95", "p95"),
    ("no_exit_continuation_regret", "No-exit mean", "mean"),
    ("no_exit_continuation_regret", "No-exit P95", "p95"),
]


def metric_value(values: np.ndarray, kind: str) -> float:
    if kind == "mean":
        return float(values.mean())
    if kind == "p95":
        return float(np.quantile(values, 0.95))
    if kind == "max":
        return float(values.max())
    raise ValueError(f"Unknown metric kind: {kind}")


def run_metric_average(samples: pd.DataFrame, column: str, kind: str) -> float:
    values = []
    for _, group in samples.groupby("run"):
        values.append(metric_value(group[column].to_numpy(dtype=float), kind))
    return float(np.mean(values))


def episode_bootstrap(samples: pd.DataFrame, column: str, kind: str, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [(run, group[column].to_numpy(dtype=float)) for run, group in samples.groupby("run")]
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        run_values = []
        for _, values in groups:
            sampled = values[rng.integers(0, len(values), len(values))]
            run_values.append(metric_value(sampled, kind))
        boot[draw] = float(np.mean(run_values))
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lccc}", "\\toprule"]
    rows.append("Metric & Estimate & MC low & MC high \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        rows.append(
            f"{row['metric']} & {row['estimate']:.3f} & {row['mc_ci_low']:.3f} & {row['mc_ci_high']:.3f} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="experiments/dynamic_continuation_audit_samples/continuation_audit_samples.csv")
    parser.add_argument("--out-csv", default="paper/tables/continuation_sample_bootstrap.csv")
    parser.add_argument("--out-tex", default="paper/tables/continuation_sample_bootstrap.tex")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    rows = []
    for idx, (column, label, kind) in enumerate(METRICS):
        estimate = run_metric_average(samples, column, kind)
        low, high = episode_bootstrap(samples, column, kind, args.draws, args.seed + idx)
        rows.append(
            {
                "metric": label,
                "column": column,
                "kind": kind,
                "estimate": estimate,
                "mc_ci_low": low,
                "mc_ci_high": high,
                "runs": int(samples["run"].nunique()),
                "samples": int(len(samples)),
            }
        )

    summary = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    write_latex_table(summary, Path(args.out_tex))
    print(summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
