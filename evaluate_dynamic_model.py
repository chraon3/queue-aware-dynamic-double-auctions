from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from dynamic_double_auction import (
    ContinuationValueNet,
    DynamicConfig,
    build_dynamic_model,
    evaluate_dynamic_baselines,
    evaluate_static_baselines,
    simulate_dynamic,
)


def load_config(path: Path) -> DynamicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = DynamicConfig.__dataclass_fields__.keys()
    return DynamicConfig(**{k: v for k, v in data.items() if k in valid})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-name", default="metrics_recheck.json")
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--regret-grid", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.json")
    updates = {}
    if args.eval_episodes:
        updates["eval_episodes"] = args.eval_episodes
    if args.regret_grid:
        updates["regret_grid"] = args.regret_grid
    cfg = replace(cfg, **updates)

    device = torch.device(cfg.device)
    model = build_dynamic_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    value_net = ContinuationValueNet(hidden=cfg.value_hidden).to(device)
    value_path = run_dir / "value_model.pt"
    if value_path.exists():
        value_net.load_state_dict(torch.load(value_path, map_location=device))
    model.eval()
    value_net.eval()
    with torch.no_grad():
        sim = simulate_dynamic(model, cfg, cfg.eval_episodes, train=True, value_net=value_net if value_path.exists() else None)
    metrics = {key: float(value.item()) for key, value in sim.items()}
    metrics.update(evaluate_static_baselines(cfg))
    metrics.update(evaluate_dynamic_baselines(cfg))
    first_best_obj = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    metrics["dynamic_neural_efficiency"] = metrics["objective"] / first_best_obj
    (run_dir / args.out_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
