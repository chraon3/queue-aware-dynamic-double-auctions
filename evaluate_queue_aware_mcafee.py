from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dynamic_double_auction import (
    DynamicConfig,
    QueueAwareMcAfeeMechanism,
    evaluate_dynamic_baselines,
    set_seed,
    simulate_dynamic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--eval-episodes", type=int, default=700)
    parser.add_argument("--regret-grid", type=int, default=7)
    parser.add_argument("--out-dir", default="experiments/queue_aware_mcafee_certified")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-buyers", type=int, default=4)
    parser.add_argument("--max-sellers", type=int, default=4)
    parser.add_argument("--wait-cost", type=float, default=0.02)
    parser.add_argument("--arrival-prob-buyer", type=float, default=0.75)
    parser.add_argument("--arrival-prob-seller", type=float, default=0.75)
    parser.add_argument("--max-patience", type=int, default=5)
    parser.add_argument("--abandon-base", type=float, default=0.01)
    parser.add_argument("--abandon-slope", type=float, default=0.08)
    args = parser.parse_args()

    cfg = DynamicConfig(
        max_buyers=args.max_buyers,
        max_sellers=args.max_sellers,
        horizon=args.horizon,
        eval_episodes=args.eval_episodes,
        regret_grid=args.regret_grid,
        wait_cost=args.wait_cost,
        arrival_prob_buyer=args.arrival_prob_buyer,
        arrival_prob_seller=args.arrival_prob_seller,
        max_patience=args.max_patience,
        abandon_base=args.abandon_base,
        abandon_slope=args.abandon_slope,
        seed=args.seed,
        mechanism="queue_mcafee",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    model = QueueAwareMcAfeeMechanism(cfg)
    with torch.no_grad():
        sim = simulate_dynamic(model, cfg, cfg.eval_episodes, train=True)
    metrics = {key: float(value.item()) for key, value in sim.items()}
    metrics.update(evaluate_dynamic_baselines(replace(cfg, mechanism="base")))
    first_best_obj = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    metrics["dynamic_queue_mcafee_certified_efficiency"] = metrics["objective"] / first_best_obj
    metrics["dynamic_queue_mcafee_certified_objective"] = metrics["objective"]
    metrics["dynamic_queue_mcafee_certified_mean_regret"] = metrics["mean_regret"]
    metrics["dynamic_queue_mcafee_certified_p95_regret"] = metrics["p95_regret"]
    metrics["dynamic_queue_mcafee_certified_max_regret"] = metrics["max_regret"]
    (out_dir / f"queue_aware_mcafee_seed{cfg.seed}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / f"config_seed{cfg.seed}.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
