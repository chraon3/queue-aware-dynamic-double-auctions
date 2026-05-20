from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from dynamic_double_auction import DynamicConfig, train_dynamic


def base_config() -> DynamicConfig:
    return DynamicConfig(
        max_buyers=4,
        max_sellers=4,
        horizon=6,
        batch_size=80,
        train_steps=130,
        eval_episodes=450,
        regret_grid=5,
        regret_method="grid",
        regret_weight=2.0,
        regret_target=0.025,
        augmented_rho=4.0,
        wait_cost=0.02,
        arrival_prob_buyer=0.75,
        arrival_prob_seller=0.75,
        max_patience=5,
        abandon_base=0.01,
        abandon_slope=0.08,
        value_loss_weight=0.2,
        feature_mode="ranked",
        log_every=50,
    )


def suite(kind: str) -> list[DynamicConfig]:
    seeds = [301, 303, 307, 311]
    base = base_config()
    configs: list[DynamicConfig] = []
    for seed in seeds:
        if kind == "no_value":
            cfg = replace(
                base,
                seed=seed,
                value_loss_weight=0.0,
                value_refine_steps=0,
                out_dir=f"experiments/dynamic_ablation_no_value_seed{seed}",
            )
        elif kind == "hybrid_regret":
            cfg = replace(
                base,
                seed=seed,
                batch_size=64,
                train_steps=110,
                regret_method="hybrid",
                adv_steps=4,
                adv_lr=0.7,
                adv_restarts=1,
                value_loss_weight=0.2,
                out_dir=f"experiments/dynamic_ablation_hybrid_regret_seed{seed}",
            )
        else:
            raise ValueError(f"Unknown ablation kind: {kind}")
        configs.append(cfg)
    return configs


def summarize(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    metrics = [
        "dynamic_neural_efficiency",
        "objective",
        "mean_regret",
        "p95_regret",
        "max_regret",
        "mean_abandonment",
        "bellman_residual",
    ]
    return raw.groupby("kind")[metrics].agg(["mean", "std"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["no_value", "hybrid_regret", "all"], default="all")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    kinds = ["no_value", "hybrid_regret"] if args.kind == "all" else [args.kind]
    Path("experiments").mkdir(exist_ok=True)
    rows = []
    for kind in kinds:
        for cfg in suite(kind):
            metrics_path = Path(cfg.out_dir) / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                print(f"Skipping existing {cfg.out_dir}", flush=True)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                print(f"\n=== Running {kind}: {cfg.out_dir} ===", flush=True)
                _, _, metrics = train_dynamic(cfg)
            row = {"kind": kind, "seed": cfg.seed, **metrics}
            rows.append(row)

    raw = pd.DataFrame(rows)
    out_dir = Path("experiments/dynamic_ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "dynamic_ablation_by_run.csv", index=False)
    summary = summarize(rows)
    if not summary.empty:
        summary.to_csv(out_dir / "dynamic_ablation_summary.csv")
        print(summary.to_string(), flush=True)
    Path("experiments/dynamic_ablation_manifest.json").write_text(
        json.dumps({"configs": [asdict(cfg) for kind in kinds for cfg in suite(kind)]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
