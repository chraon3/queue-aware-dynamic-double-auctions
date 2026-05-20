from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = [
    ("Full", "experiments/dynamic_continuation_audit/continuation_audit_by_run.csv"),
    ("Tail", "experiments/dynamic_continuation_audit_tail/continuation_audit_by_run.csv"),
    ("Hard-tail", "experiments/dynamic_continuation_audit_hardcase_select/continuation_audit_by_run.csv"),
]

METRICS = [
    ("continuation_mean_regret", "Mean"),
    ("continuation_p95_regret", "P95"),
    ("continuation_max_regret", "Max"),
]


def seed_from_run(run: str) -> int:
    match = re.search(r"seed(\d+)", run)
    if match is None:
        raise ValueError(f"Cannot parse seed from run name: {run}")
    return int(match.group(1))


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        boot[draw] = values[rng.integers(0, n, n)].mean()
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def read_variant(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["seed"] = frame["run"].map(seed_from_run)
    return frame.set_index("seed").sort_index()


def interval_cell(mean: float, low: float, high: float) -> str:
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def write_level_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lccc}", "\\toprule"]
    rows.append("Variant & Mean & P95 & Max \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["variant"]),
                    interval_cell(row["mean"], row["mean_ci_low"], row["mean_ci_high"]),
                    interval_cell(row["p95"], row["p95_ci_low"], row["p95_ci_high"]),
                    interval_cell(row["max"], row["max_ci_low"], row["max_ci_high"]),
                ]
            )
            + " \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_delta_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lccccc}", "\\toprule"]
    rows.append("Variant & dMean & dP95 & dMax & P95 wins & Max wins \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["variant"]),
                    interval_cell(row["dmean"], row["dmean_ci_low"], row["dmean_ci_high"]),
                    interval_cell(row["dp95"], row["dp95_ci_low"], row["dp95_ci_high"]),
                    interval_cell(row["dmax"], row["dmax_ci_low"], row["dmax_ci_high"]),
                    f"{int(row['p95_wins'])}/{int(row['seeds'])}",
                    f"{int(row['max_wins'])}/{int(row['seeds'])}",
                ]
            )
            + " \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--out-dir", default="paper/tables")
    args = parser.parse_args()

    frames = {label: read_variant(path) for label, path in VARIANTS}
    level_rows = []
    for v_idx, (label, _) in enumerate(VARIANTS):
        row: dict[str, float | str | int] = {"variant": label, "seeds": len(frames[label])}
        for m_idx, (metric, short) in enumerate(METRICS):
            values = frames[label][metric].to_numpy(dtype=float)
            low, high = bootstrap_mean(values, args.draws, args.seed + 10 * v_idx + m_idx)
            key = short.lower()
            row[key] = float(values.mean())
            row[f"{key}_ci_low"] = low
            row[f"{key}_ci_high"] = high
        level_rows.append(row)
    level = pd.DataFrame(level_rows)

    full = frames["Full"]
    delta_rows = []
    for v_idx, label in enumerate(["Tail", "Hard-tail"]):
        candidate = frames[label]
        common = full.index.intersection(candidate.index)
        row = {"variant": f"{label} - Full", "seeds": len(common)}
        for m_idx, (metric, short) in enumerate(METRICS):
            delta = (candidate.loc[common, metric] - full.loc[common, metric]).to_numpy(dtype=float)
            low, high = bootstrap_mean(delta, args.draws, args.seed + 100 + 10 * v_idx + m_idx)
            key = f"d{short.lower()}"
            row[key] = float(delta.mean())
            row[f"{key}_ci_low"] = low
            row[f"{key}_ci_high"] = high
            row[f"{short.lower()}_wins"] = int((delta < 0.0).sum())
        delta_rows.append(row)
    delta = pd.DataFrame(delta_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    level.to_csv(out_dir / "continuation_bootstrap.csv", index=False)
    delta.to_csv(out_dir / "continuation_paired_bootstrap.csv", index=False)
    write_level_table(level, out_dir / "continuation_bootstrap.tex")
    write_delta_table(delta, out_dir / "continuation_paired_bootstrap.tex")
    print(
        {
            "levels": level.to_dict(orient="records"),
            "paired_deltas": delta.to_dict(orient="records"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
