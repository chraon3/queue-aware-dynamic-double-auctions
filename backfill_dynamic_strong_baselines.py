from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from dynamic_double_auction import DynamicConfig, evaluate_dynamic_baselines


def load_config(path: Path) -> DynamicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = DynamicConfig.__dataclass_fields__.keys()
    return DynamicConfig(**{key: value for key, value in data.items() if key in valid})


def discover_runs(patterns: Iterable[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        runs.extend(sorted(Path().glob(pattern)))
    return [path for path in runs if (path / "config.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--metric-name", default="metrics_recheck.json")
    parser.add_argument("--fallback-name", default="metrics.json")
    parser.add_argument("--eval-episodes", type=int, default=700)
    args = parser.parse_args()

    runs = discover_runs(args.runs)
    if not runs:
        raise FileNotFoundError("No dynamic run directories found.")

    for run_dir in runs:
        metric_path = run_dir / args.metric_name
        if not metric_path.exists():
            metric_path = run_dir / args.fallback_name
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            metrics = {}

        cfg = replace(load_config(run_dir / "config.json"), eval_episodes=args.eval_episodes)
        baseline_metrics = evaluate_dynamic_baselines(cfg)
        metrics.update(baseline_metrics)
        first_best = max(metrics["dynamic_first_best_objective"], 1.0e-9)
        if "objective" in metrics:
            metrics["dynamic_neural_efficiency"] = metrics["objective"] / first_best

        out_path = run_dir / args.metric_name
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"{run_dir.name}: state_posted={metrics['dynamic_state_posted_efficiency']:.3f} "
            f"queue_mcafee={metrics['dynamic_queue_mcafee_efficiency']:.3f} "
            f"neural={metrics.get('dynamic_neural_efficiency', float('nan')):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
