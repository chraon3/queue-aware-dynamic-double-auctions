from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARKS = [
    ("Dynamic McAfee", "McAfee"),
    ("Dynamic posted price", "Posted"),
    ("Optimized state posted price", "State posted"),
    ("Dynamic trade reduction", "Trade reduction"),
    ("Queue-aware McAfee", "Queue-aware"),
]


def paired_bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    n = len(values)
    for draw in range(draws):
        boot[draw] = values[rng.integers(0, n, n)].mean()
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule"]
    rows.append("Benchmark & Mean diff. & CI low & CI high & Wins & Min diff. \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        values = [
            str(row["benchmark"]),
            f"{row['mean_diff']:.3f}",
            f"{row['ci_low']:.3f}",
            f"{row['ci_high']:.3f}",
            f"{int(row['wins'])}/{int(row['seeds'])}",
            f"{row['min_diff']:.3f}",
        ]
        rows.append(" & ".join(values) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-seed", default="paper/tables/dynamic_results_by_seed.csv")
    parser.add_argument("--out-csv", default="paper/tables/dynamic_paired_bootstrap.csv")
    parser.add_argument("--out-tex", default="paper/tables/dynamic_paired_bootstrap.tex")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data = pd.read_csv(args.by_seed)
    wide = data.pivot(index="run", columns="policy", values="efficiency")
    neural = wide["Neural dynamic mechanism"]
    rows = []
    for idx, (policy, label) in enumerate(BENCHMARKS):
        diff = (neural - wide[policy]).dropna().to_numpy(dtype=float)
        ci_low, ci_high = paired_bootstrap(diff, args.draws, args.seed + idx)
        rows.append(
            {
                "benchmark": label,
                "seeds": len(diff),
                "mean_diff": float(diff.mean()),
                "std_diff": float(diff.std(ddof=1)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wins": int((diff > 0).sum()),
                "min_diff": float(diff.min()),
                "max_diff": float(diff.max()),
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
