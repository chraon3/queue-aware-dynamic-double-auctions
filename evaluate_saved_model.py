from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from econpinn_double_auction import Config, HardConstrainedDoubleAuction, evaluate


def load_config(path: Path) -> Config:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid_keys = Config.__dataclass_fields__.keys()
    return Config(**{key: value for key, value in data.items() if key in valid_keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-name", type=str, default="metrics_eval.json")
    parser.add_argument("--eval-regret-method", choices=["grid", "dense", "adv", "hybrid", "exact", "audit"], default="")
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--regret-eval-samples", type=int, default=0)
    parser.add_argument("--eval-grid", type=int, default=0)
    parser.add_argument("--dense-grid", type=int, default=0)
    parser.add_argument("--adv-steps", type=int, default=0)
    parser.add_argument("--adv-restarts", type=int, default=0)
    parser.add_argument("--exact-maxiter", type=int, default=0)
    parser.add_argument("--exact-popsize", type=int, default=0)
    parser.add_argument("--exact-tol", type=float, default=0.0)
    parser.add_argument(
        "--distribution",
        choices=["uniform", "beta_easy", "beta_hard", "asymmetric", "correlated"],
        default="",
    )
    parser.add_argument("--clearance-cost", type=float, default=-1.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.json")
    updates = {}
    if args.eval_regret_method:
        updates["eval_regret_method"] = args.eval_regret_method
    if args.eval_samples:
        updates["eval_samples"] = args.eval_samples
    if args.regret_eval_samples:
        updates["regret_eval_samples"] = args.regret_eval_samples
    if args.eval_grid:
        updates["eval_grid"] = args.eval_grid
    if args.dense_grid:
        updates["dense_grid"] = args.dense_grid
    if args.adv_steps:
        updates["adv_steps"] = args.adv_steps
    if args.adv_restarts:
        updates["adv_restarts"] = args.adv_restarts
    if args.exact_maxiter:
        updates["exact_maxiter"] = args.exact_maxiter
    if args.exact_popsize:
        updates["exact_popsize"] = args.exact_popsize
    if args.exact_tol:
        updates["exact_tol"] = args.exact_tol
    if args.distribution:
        updates["distribution"] = args.distribution
    if args.clearance_cost >= 0.0:
        updates["clearance_cost"] = args.clearance_cost
    cfg = replace(cfg, **updates)

    device = torch.device(cfg.device)
    model = HardConstrainedDoubleAuction(
        hidden=cfg.hidden,
        depth=cfg.depth,
        feature_mode=cfg.feature_mode,
        clearance_cost=cfg.clearance_cost,
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    metrics = evaluate(model, cfg, device)

    (run_dir / args.out_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / f"{Path(args.out_name).stem}_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
