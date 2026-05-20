from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch

from dynamic_double_auction import (
    ContinuationValueNet,
    DynamicConfig,
    build_dynamic_model,
    evaluate_dynamic_baselines,
    set_seed,
    simulate_dynamic,
)
from torch import nn


SCENARIOS: Dict[str, Dict[str, float | int]] = {
    "in_sample": {},
    "buyer_congested": {"arrival_prob_buyer": 0.90, "arrival_prob_seller": 0.60},
    "seller_congested": {"arrival_prob_buyer": 0.60, "arrival_prob_seller": 0.90},
    "high_wait_cost": {"wait_cost": 0.04},
    "impatient": {"max_patience": 3, "abandon_slope": 0.16},
    "thin_market": {"arrival_prob_buyer": 0.55, "arrival_prob_seller": 0.55},
}

SCENARIO_LABELS = {
    "in_sample": "In sample",
    "buyer_congested": "Buyer congested",
    "seller_congested": "Seller congested",
    "high_wait_cost": "High wait cost",
    "impatient": "Impatient agents",
    "thin_market": "Thin market",
}


def load_config(path: Path) -> DynamicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = DynamicConfig.__dataclass_fields__.keys()
    return DynamicConfig(**{key: value for key, value in data.items() if key in valid})


def load_run(run_dir: Path) -> tuple[DynamicConfig, nn.Module, ContinuationValueNet | None]:
    cfg = load_config(run_dir / "config.json")
    device = torch.device(cfg.device)
    model = build_dynamic_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    value_net = None
    value_path = run_dir / "value_model.pt"
    if value_path.exists():
        value_net = ContinuationValueNet(hidden=cfg.value_hidden).to(device)
        value_net.load_state_dict(torch.load(value_path, map_location=device))
        value_net.eval()
    return cfg, model, value_net


def evaluate_run_scenario(
    run_dir: Path,
    scenario: str,
    updates: Dict[str, float | int],
    eval_episodes: int,
    regret_grid: int,
    seed_offset: int,
) -> Dict[str, float | str]:
    cfg, model, value_net = load_run(run_dir)
    cfg_eval = replace(
        cfg,
        eval_episodes=eval_episodes,
        regret_grid=regret_grid,
        **updates,
    )
    set_seed(cfg.seed + seed_offset)
    with torch.no_grad():
        sim = simulate_dynamic(model, cfg_eval, cfg_eval.eval_episodes, train=True, value_net=value_net)
    metrics = {key: float(value.item()) for key, value in sim.items()}
    metrics.update(evaluate_dynamic_baselines(cfg_eval))
    first_best = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    metrics["dynamic_neural_efficiency"] = metrics["objective"] / first_best
    return {
        "run": run_dir.name,
        "scenario": scenario,
        "label": SCENARIO_LABELS[scenario],
        "neural_efficiency": metrics["dynamic_neural_efficiency"],
        "neural_objective": metrics["objective"],
        "mcafee_efficiency": metrics["dynamic_mcafee_efficiency"],
        "posted_efficiency": metrics["dynamic_posted_efficiency"],
        "trade_reduction_efficiency": metrics["dynamic_trade_reduction_efficiency"],
        "mean_regret": metrics["mean_regret"],
        "p95_regret": metrics["p95_regret"],
        "max_regret": metrics["max_regret"],
        "bellman_residual": metrics["bellman_residual"],
        "abandonment": metrics["mean_abandonment"],
        "first_best_objective": metrics["dynamic_first_best_objective"],
    }


def discover_runs(patterns: Iterable[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        runs.extend(sorted(Path().glob(pattern)))
    return [path for path in runs if (path / "config.json").exists() and (path / "model.pt").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--regret-grid", type=int, default=7)
    parser.add_argument("--out-dir", default="experiments/dynamic_robustness")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    runs = discover_runs(args.runs)
    if not runs:
        raise FileNotFoundError("No dynamic runs found. Expected config.json and model.pt under the run directories.")

    for run_idx, run_dir in enumerate(runs):
        for scenario_idx, (scenario, updates) in enumerate(SCENARIOS.items()):
            print(f"Evaluating {run_dir.name} / {scenario}", flush=True)
            row = evaluate_run_scenario(
                run_dir,
                scenario,
                updates,
                args.eval_episodes,
                args.regret_grid,
                seed_offset=10_000 + 997 * run_idx + 37 * scenario_idx,
            )
            rows.append(row)
            (out_dir / f"{run_dir.name}__{scenario}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    raw = pd.DataFrame(rows)
    raw["scenario_order"] = raw["scenario"].map({name: idx for idx, name in enumerate(SCENARIOS)})
    raw = raw.sort_values(["scenario_order", "run"]).drop(columns=["scenario_order"])
    raw.to_csv(out_dir / "robustness_by_run.csv", index=False)

    summary = raw.groupby(["scenario", "label"], sort=False).agg(
        neural_efficiency=("neural_efficiency", "mean"),
        neural_efficiency_std=("neural_efficiency", "std"),
        mcafee_efficiency=("mcafee_efficiency", "mean"),
        posted_efficiency=("posted_efficiency", "mean"),
        trade_reduction_efficiency=("trade_reduction_efficiency", "mean"),
        mean_regret=("mean_regret", "mean"),
        p95_regret=("p95_regret", "mean"),
        abandonment=("abandonment", "mean"),
    ).reset_index()
    summary["scenario_order"] = summary["scenario"].map({name: idx for idx, name in enumerate(SCENARIOS)})
    summary = summary.sort_values("scenario_order").drop(columns=["scenario_order"])
    summary.to_csv(out_dir / "robustness_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
