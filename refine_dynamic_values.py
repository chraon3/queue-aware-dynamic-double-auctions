from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from dynamic_double_auction import (
    ContinuationValueNet,
    DynamicConfig,
    evaluate_dynamic_baselines,
    evaluate_static_baselines,
    refine_value_net,
    set_seed,
    simulate_dynamic,
)
from econpinn_double_auction import HardConstrainedDoubleAuction


def load_config(path: Path) -> DynamicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = DynamicConfig.__dataclass_fields__.keys()
    return DynamicConfig(**{key: value for key, value in data.items() if key in valid})


def refine_run(
    run_dir: Path,
    steps: int,
    batch_size: int,
    lr: float,
    td_weight: float,
    log_every: int,
    eval_episodes: int,
    regret_grid: int,
    out_name: str,
    seed_offset: int,
    merge_value_only: bool,
) -> dict[str, float]:
    cfg = load_config(run_dir / "config.json")
    cfg = replace(
        cfg,
        value_refine_steps=steps,
        value_refine_batch_size=batch_size,
        value_refine_lr=lr,
        value_refine_td_weight=td_weight,
        value_refine_log_every=log_every,
        eval_episodes=eval_episodes,
        regret_grid=regret_grid,
    )
    set_seed(cfg.seed + seed_offset)
    device = torch.device(cfg.device)

    model = HardConstrainedDoubleAuction(cfg.hidden, cfg.depth, cfg.feature_mode).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    value_net = ContinuationValueNet(hidden=cfg.value_hidden).to(device)
    value_path = run_dir / "value_model.pt"
    if value_path.exists():
        value_net.load_state_dict(torch.load(value_path, map_location=device))

    refine_value_net(model, value_net, cfg, run_dir)
    torch.save(value_net.state_dict(), run_dir / "value_model.pt")
    (run_dir / "value_refine_config.json").write_text(json.dumps({
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "td_weight": td_weight,
        "eval_episodes": eval_episodes,
        "regret_grid": regret_grid,
        "seed_offset": seed_offset,
    }, indent=2), encoding="utf-8")

    model.eval()
    value_net.eval()
    set_seed(cfg.seed + seed_offset + 10_000)
    out_path = run_dir / out_name
    value_only_eval = merge_value_only and out_path.exists()
    with torch.no_grad():
        sim = simulate_dynamic(model, cfg, cfg.eval_episodes, train=not value_only_eval, value_net=value_net)
    metrics = {key: float(value.item()) for key, value in sim.items()}
    if value_only_eval:
        base_metrics = json.loads(out_path.read_text(encoding="utf-8"))
        for key in [
            "bellman_residual",
            "pathwise_bellman_residual",
            "value_loss",
            "value_mae",
            "value_rmse",
        ]:
            base_metrics[key] = metrics[key]
        base_metrics["value_refined"] = True
        metrics = base_metrics
    else:
        metrics.update(evaluate_static_baselines(cfg))
        metrics.update(evaluate_dynamic_baselines(cfg))
        first_best_obj = max(metrics["dynamic_first_best_objective"], 1.0e-9)
        metrics["dynamic_neural_efficiency"] = metrics["objective"] / first_best_obj
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def discover_runs(patterns: list[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        runs.extend(sorted(Path().glob(pattern)))
    return [path for path in runs if (path / "model.pt").exists() and (path / "config.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--td-weight", type=float, default=0.75)
    parser.add_argument("--log-every", type=int, default=70)
    parser.add_argument("--eval-episodes", type=int, default=700)
    parser.add_argument("--regret-grid", type=int, default=7)
    parser.add_argument("--out-name", default="metrics_recheck.json")
    parser.add_argument("--replace-policy-metrics", action="store_true", help="Replace all metrics instead of preserving existing policy/regret metrics and updating only value diagnostics.")
    args = parser.parse_args()

    rows = []
    for idx, run_dir in enumerate(discover_runs(args.runs)):
        print(f"\n=== Refining {run_dir} ===", flush=True)
        metrics = refine_run(
            run_dir,
            args.steps,
            args.batch_size,
            args.lr,
            args.td_weight,
            args.log_every,
            args.eval_episodes,
            args.regret_grid,
            args.out_name,
            seed_offset=70_000 + 997 * idx,
            merge_value_only=not args.replace_policy_metrics,
        )
        rows.append({"run": run_dir.name, **metrics})
        print(
            f"{run_dir.name}: td={metrics['bellman_residual']:.5f} "
            f"value_loss={metrics['value_loss']:.5f} eff={metrics['dynamic_neural_efficiency']:.3f}",
            flush=True,
        )
    Path("experiments/value_refine_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
