from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch

from dynamic_continuation_audit import audit_run, parse_exit_thresholds
from dynamic_double_auction import (
    build_dynamic_model,
    evaluate_dynamic_baselines,
    evaluate_static_baselines,
    simulate_dynamic,
)
from dynamic_history_audit import load_config


DELTA_COLUMNS = [
    "buyer_continuation_mean_regret",
    "seller_continuation_mean_regret",
    "continuation_mean_regret",
    "continuation_p95_regret",
    "continuation_max_regret",
    "no_exit_continuation_mean_regret",
    "no_exit_continuation_p95_regret",
    "no_exit_continuation_max_regret",
]


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one alpha is required.")
    return values


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed offset is required.")
    return values


def load_state(run_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    return torch.load(run_dir / "model.pt", map_location=device)


def alpha_name(alpha: float) -> str:
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return "alpha_" + text.replace("-", "m").replace(".", "p")


def save_interpolated_run(base_run: Path, candidate_run: Path, out_dir: Path, alpha: float, eval_episodes: int) -> dict[str, float]:
    cfg = load_config(base_run / "config.json")
    device = torch.device(cfg.device)
    base_state = load_state(base_run, device)
    candidate_state = load_state(candidate_run, device)
    if set(base_state) != set(candidate_state):
        missing = sorted(set(base_state).symmetric_difference(candidate_state))
        raise ValueError(f"State-dict keys differ between runs: {missing[:10]}")
    mixed_state: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        candidate_value = candidate_state[key]
        if base_value.shape != candidate_value.shape:
            raise ValueError(f"State tensor shape differs for {key}: {base_value.shape} vs {candidate_value.shape}")
        if torch.is_floating_point(base_value):
            mixed_state[key] = (1.0 - alpha) * base_value + alpha * candidate_value
        else:
            mixed_state[key] = candidate_value.clone()

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, out_dir=str(out_dir), eval_episodes=eval_episodes)
    model = build_dynamic_model(cfg).to(device)
    model.load_state_dict(mixed_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    manifest = {
        "base_run": str(base_run),
        "candidate_run": str(candidate_run),
        "alpha": alpha,
    }
    (out_dir / "interpolation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    model.eval()
    with torch.no_grad():
        sim = simulate_dynamic(model, cfg, eval_episodes, train=True)
    metrics = {key: float(value.item()) for key, value in sim.items()}
    metrics.update(evaluate_static_baselines(cfg))
    metrics.update(evaluate_dynamic_baselines(cfg))
    first_best = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    metrics["dynamic_neural_efficiency"] = metrics["objective"] / first_best
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--seed-offsets", default="170000,190000")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=220)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--family", choices=["age", "state", "history"], default="history")
    parser.add_argument("--history-draws", type=int, default=8)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--mean-tolerance", type=float, default=0.0)
    parser.add_argument("--p95-tolerance", type=float, default=0.0)
    parser.add_argument("--max-tolerance", type=float, default=0.0)
    parser.add_argument("--min-efficiency", type=float, default=0.0)
    args = parser.parse_args()

    base_run = Path(args.base_run)
    candidate_run = Path(args.candidate_run)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    alphas = parse_floats(args.alphas)
    offsets = parse_ints(args.seed_offsets)
    exit_thresholds = parse_exit_thresholds(args.exit_thresholds)

    base_rows = {}
    for offset in offsets:
        row, _details = audit_run(
            base_run,
            args.episodes,
            args.grid,
            args.radius,
            args.family,
            args.history_draws,
            exit_thresholds,
            args.chunk_size,
            offset,
            0,
        )
        base_rows[offset] = row

    run_rows: list[dict[str, float | str]] = []
    delta_rows: list[dict[str, float | str | bool]] = []
    for alpha in alphas:
        alpha_dir = out_root / alpha_name(alpha)
        metrics = save_interpolated_run(base_run, candidate_run, alpha_dir, alpha, args.eval_episodes)
        for offset in offsets:
            row, _details = audit_run(
                alpha_dir,
                args.episodes,
                args.grid,
                args.radius,
                args.family,
                args.history_draws,
                exit_thresholds,
                args.chunk_size,
                offset,
                0,
            )
            row["alpha"] = alpha
            row["paired_offset"] = float(offset)
            run_rows.append(row)
            delta_row: dict[str, float | str | bool] = {
                "alpha": alpha,
                "paired_offset": float(offset),
                "baseline_run": base_run.name,
                "candidate_run": alpha_dir.name,
                "objective": metrics["objective"],
                "dynamic_neural_efficiency": metrics["dynamic_neural_efficiency"],
            }
            for column in DELTA_COLUMNS:
                delta_row[f"delta_{column}"] = float(row[column]) - float(base_rows[offset][column])
            delta_rows.append(delta_row)

    run_table = pd.DataFrame(run_rows)
    delta_table = pd.DataFrame(delta_rows)
    run_table.to_csv(out_root / "interpolation_continuation_by_run.csv", index=False)
    delta_table.to_csv(out_root / "interpolation_continuation_deltas.csv", index=False)
    grouped = delta_table.groupby("alpha")
    summary_rows: list[dict[str, float | bool]] = []
    for alpha, group in grouped:
        row: dict[str, float | bool] = {"alpha": float(alpha), "offset_count": float(len(group))}
        for column in DELTA_COLUMNS:
            delta_column = f"delta_{column}"
            row[f"mean_{delta_column}"] = float(group[delta_column].mean())
            row[f"max_{delta_column}"] = float(group[delta_column].max())
            row[f"wins_{delta_column}"] = float((group[delta_column] < 0.0).sum())
        row["objective"] = float(group["objective"].mean())
        row["dynamic_neural_efficiency"] = float(group["dynamic_neural_efficiency"].mean())
        row["screened_adopted"] = bool(
            row["mean_delta_continuation_mean_regret"] <= args.mean_tolerance
            and row["mean_delta_continuation_p95_regret"] <= args.p95_tolerance
            and row["mean_delta_continuation_max_regret"] <= args.max_tolerance
            and row["dynamic_neural_efficiency"] >= args.min_efficiency
        )
        row["screen_score"] = float(
            row["mean_delta_continuation_mean_regret"]
            + row["mean_delta_continuation_p95_regret"]
            + row["mean_delta_continuation_max_regret"]
        )
        summary_rows.append(row)

    summary_table = pd.DataFrame(summary_rows).sort_values(["screened_adopted", "screen_score"], ascending=[False, True])
    summary_table.to_csv(out_root / "interpolation_summary.csv", index=False)
    summary = {
        "base_run": str(base_run),
        "candidate_run": str(candidate_run),
        "alphas": alphas,
        "offsets": offsets,
        "screening_rule": {
            "mean_tolerance": args.mean_tolerance,
            "p95_tolerance": args.p95_tolerance,
            "max_tolerance": args.max_tolerance,
            "min_efficiency": args.min_efficiency,
        },
        "best_alpha": float(summary_table.iloc[0]["alpha"]) if not summary_table.empty else None,
        "summary": summary_table.to_dict(orient="records"),
    }
    (out_root / "interpolation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
