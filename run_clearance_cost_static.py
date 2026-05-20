from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from econpinn_double_auction import Config, train, write_report


def configs(args: argparse.Namespace) -> list[Config]:
    base = Config(
        n_buyers=5,
        n_sellers=3,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_samples=args.eval_samples,
        regret_eval_samples=args.regret_eval_samples,
        regret_grid=args.regret_grid,
        eval_grid=args.eval_grid,
        hidden=64,
        depth=3,
        lr=2.0e-3,
        feature_mode="ranked",
        train_method="dual",
        regret_method="grid",
        eval_regret_method="hybrid",
        regret_weight=2.0,
        regret_target=args.regret_target,
        dual_lr=0.8,
        distribution="asymmetric",
        clearance_cost=args.clearance_cost,
        log_every=max(args.train_steps // 4, 1),
    )
    return [
        replace(base, seed=seed, out_dir=str(Path(args.out_root) / f"clearance_asym_5x3_seed{seed}"))
        for seed in args.seeds
    ]


def load_metrics(run_dir: Path) -> dict[str, float | str]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def row_from_metrics(cfg: Config, metrics: dict[str, float | str]) -> dict[str, float | str | int]:
    return {
        "seed": cfg.seed,
        "run": Path(cfg.out_dir).name,
        "n_buyers": cfg.n_buyers,
        "n_sellers": cfg.n_sellers,
        "distribution": cfg.distribution,
        "clearance_cost": cfg.clearance_cost,
        "neural_welfare": float(metrics["neural_welfare"]),
        "first_best_welfare": float(metrics["first_best_welfare"]),
        "mcafee_welfare": float(metrics["mcafee_welfare"]),
        "posted_price_welfare": float(metrics["posted_price_welfare"]),
        "trade_reduction_welfare": float(metrics["trade_reduction_welfare"]),
        "neural_efficiency": float(metrics["neural_efficiency"]),
        "mcafee_efficiency": float(metrics["mcafee_efficiency"]),
        "posted_price_efficiency": float(metrics["posted_price_efficiency"]),
        "trade_reduction_efficiency": float(metrics["trade_reduction_efficiency"]),
        "total_mean_regret": float(metrics["mean_buyer_regret"]) + float(metrics["mean_seller_regret"]),
        "p95_regret": float(metrics["p95_total_agent_regret"]),
        "max_regret": max(float(metrics["max_buyer_regret"]), float(metrics["max_seller_regret"])),
        "mean_budget_surplus": float(metrics["mean_budget_surplus"]),
        "min_budget_surplus": float(metrics["min_budget_surplus"]),
        "max_clearing_abs": float(metrics["max_clearing_abs"]),
        "max_row_excess": float(metrics["max_row_excess"]),
        "max_col_excess": float(metrics["max_col_excess"]),
        "min_buyer_utility": float(metrics["min_buyer_utility"]),
        "min_seller_utility": float(metrics["min_seller_utility"]),
        "mean_trade_volume": float(metrics["mean_trade_volume"]),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "neural_efficiency",
        "mcafee_efficiency",
        "posted_price_efficiency",
        "trade_reduction_efficiency",
        "neural_welfare",
        "mcafee_welfare",
        "total_mean_regret",
        "p95_regret",
        "max_regret",
        "mean_budget_surplus",
        "max_clearing_abs",
        "min_buyer_utility",
        "min_seller_utility",
    ]
    row: dict[str, float | str | int] = {"environment": "Clearance-cost asymmetric 5x3", "n_seeds": int(raw["seed"].nunique())}
    for metric in metrics:
        row[metric] = float(raw[metric].mean())
        row[f"{metric}_std"] = float(raw[metric].std(ddof=1))
    row["neural_minus_mcafee_efficiency"] = float((raw["neural_efficiency"] - raw["mcafee_efficiency"]).mean())
    row["neural_wins_vs_mcafee"] = int((raw["neural_efficiency"] > raw["mcafee_efficiency"]).sum())
    return pd.DataFrame([row])


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    row = summary.iloc[0]

    def fmt(value: float) -> str:
        if pd.isna(value):
            return "--"
        return f"{value:.3f}"

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Environment & Neural & McAfee & Posted & Regret & Wins \\\\",
        "\\midrule",
        (
            f"Clearance 5x3 & {fmt(row['neural_efficiency'])} & "
            f"{fmt(row['mcafee_efficiency'])} & {fmt(row['posted_price_efficiency'])} & "
            f"{fmt(row['total_mean_regret'])} & "
            f"{int(row['neural_wins_vs_mcafee'])}/{int(row['n_seeds'])} \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[131, 137, 139])
    parser.add_argument("--out-root", default="experiments/clearance_cost_static")
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=1600)
    parser.add_argument("--regret-eval-samples", type=int, default=350)
    parser.add_argument("--regret-grid", type=int, default=7)
    parser.add_argument("--eval-grid", type=int, default=21)
    parser.add_argument("--regret-target", type=float, default=0.012)
    parser.add_argument("--clearance-cost", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--paper-table", default="paper/tables/clearance_cost_static.tex")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    rows = []
    for cfg in configs(args):
        run_dir = Path(cfg.out_dir)
        metrics_path = run_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            metrics = load_metrics(run_dir)
        else:
            print(f"Training {run_dir}", flush=True)
            _, _, metrics = train(cfg)
            write_report(cfg, metrics)
        manifest.append({"config": asdict(cfg), "metrics": metrics})
        rows.append(row_from_metrics(cfg, metrics))

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(out_root / "clearance_cost_static_by_seed.csv", index=False)
    summary.to_csv(out_root / "clearance_cost_static_summary.csv", index=False)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    table_path = Path(args.paper_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_latex(summary, table_path)
    summary.to_csv(table_path.with_suffix(".csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
