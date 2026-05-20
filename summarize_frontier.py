from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUNS = [
    ("lambda=2", Path("outputs_w2")),
    ("lambda=8", Path("outputs_main")),
    ("lambda=32", Path("outputs_w32")),
]


def discover_runs(root: Path) -> list[tuple[str, Path]]:
    runs = []
    for metrics_path in sorted(root.glob("**/metrics*.json")):
        if metrics_path.name.endswith("_config.json"):
            continue
        suffix = "" if metrics_path.stem == "metrics" else f"__{metrics_path.stem.removeprefix('metrics_')}"
        runs.append((f"{metrics_path.parent.name}{suffix}", metrics_path))
    return runs


def summarize(runs: list[tuple[str, Path]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for label, path in runs:
        metrics_path = path if path.is_file() else path / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "neural_efficiency" not in metrics:
            continue
        rows.append(
            {
                "run": label,
                "regret_method": metrics.get("regret_method", "grid"),
                "neural_efficiency": metrics["neural_efficiency"],
                "trade_reduction_efficiency": metrics["trade_reduction_efficiency"],
                "mcafee_efficiency": metrics.get("mcafee_efficiency", metrics["trade_reduction_efficiency"]),
                "posted_price_efficiency": metrics.get("posted_price_efficiency", float("nan")),
                "total_mean_regret": metrics["mean_buyer_regret"] + metrics["mean_seller_regret"],
                "mean_trade_volume": metrics["mean_trade_volume"],
                "neural_welfare": metrics["neural_welfare"],
                "first_best_welfare": metrics["first_best_welfare"],
                "trade_reduction_welfare": metrics["trade_reduction_welfare"],
                "mcafee_welfare": metrics.get("mcafee_welfare", metrics["trade_reduction_welfare"]),
                "posted_price_welfare": metrics.get("posted_price_welfare", float("nan")),
                "min_budget_surplus": metrics["min_budget_surplus"],
                "max_clearing_abs": metrics["max_clearing_abs"],
                "min_buyer_utility": metrics["min_buyer_utility"],
                "min_seller_utility": metrics["min_seller_utility"],
            }
        )

    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "frontier_summary.csv", index=False)
    if df.empty:
        return df

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    for method, group in df.groupby("regret_method"):
        ax.scatter(group["total_mean_regret"], group["neural_efficiency"], s=80, label=f"neural ({method})")
        for _, row in group.iterrows():
            ax.annotate(row["run"], (row["total_mean_regret"], row["neural_efficiency"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.axhline(df["trade_reduction_efficiency"].mean(), color="#991b1b", linestyle="--", linewidth=1.4, label="trade reduction mean")
    ax.axhline(df["mcafee_efficiency"].mean(), color="#ea580c", linestyle=":", linewidth=1.4, label="McAfee mean")
    ax.set_xscale("log")
    ax.set_xlabel("Total mean grid regret")
    ax.set_ylabel("Efficiency relative to first best")
    ax.set_title("Welfare-IC frontier prototype")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "frontier.png")
    plt.close(fig)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="outputs_summary")
    args = parser.parse_args()
    runs = discover_runs(Path(args.root)) if args.root else DEFAULT_RUNS
    summarize(runs, Path(args.out_dir))


if __name__ == "__main__":
    main()
