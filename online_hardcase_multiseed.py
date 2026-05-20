from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


KEY_DELTAS = [
    "continuation_mean_regret",
    "continuation_p95_regret",
    "continuation_max_regret",
    "no_exit_continuation_mean_regret",
    "no_exit_continuation_p95_regret",
    "no_exit_continuation_max_regret",
]


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def run_command(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_seed_summary(seed: int, out_dir: Path, base_run: Path, repaired_run: Path) -> dict[str, Any]:
    repair_summary = read_json(out_dir / "online_hardcase_summary.json")
    metrics = read_json(repaired_run / "metrics.json")
    manifest = read_json(repaired_run / "finetune_manifest.json")
    continuation_objective = manifest.get("continuation_objective", {}) if isinstance(manifest, dict) else {}
    sweep_summary = read_json(out_dir / "paired_sweep" / "paired_continuation_sweep_summary.json")
    delta_path = out_dir / "paired_sweep" / "paired_continuation_deltas.csv"
    delta_rows = read_csv_rows(delta_path)

    row: dict[str, Any] = {
        "seed": seed,
        "base_run": str(base_run),
        "repaired_run": str(repaired_run),
        "repair_objective": metrics.get("objective"),
        "repair_efficiency": metrics.get("dynamic_neural_efficiency"),
        "repair_mean_regret": metrics.get("mean_regret"),
        "repair_p95_regret": metrics.get("p95_regret"),
        "repair_max_regret": metrics.get("max_regret"),
        "best_step": continuation_objective.get("best_step"),
        "best_validation_score": continuation_objective.get("best_validation_score"),
        "include_baseline_selection": continuation_objective.get("include_baseline_selection"),
        "fixed_validation_online_hard": continuation_objective.get("fixed_validation_online_hard"),
        "offset_count": len(delta_rows),
    }
    wins = sweep_summary.get("negative_delta_wins", {}) if isinstance(sweep_summary, dict) else {}
    for metric in KEY_DELTAS:
        delta_column = f"delta_{metric}"
        values = [float(item[delta_column]) for item in delta_rows if item.get(delta_column) not in (None, "")]
        row[f"mean_delta_{metric}"] = sum(values) / len(values) if values else None
        row[f"min_delta_{metric}"] = min(values) if values else None
        row[f"max_delta_{metric}"] = max(values) if values else None
        row[f"wins_{metric}"] = wins.get(metric)
    row["repair_summary_path"] = str(out_dir / "online_hardcase_summary.json") if repair_summary else ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="81,83,89,91,93,97,101,103,107,109")
    parser.add_argument("--base-template", default="experiments/dynamic_patience_value_seed{seed}")
    parser.add_argument("--out-root", default="experiments/online_hardcase_multiseed_screen")
    parser.add_argument("--seed-base", type=int, default=950_000)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--offsets", default="170000,190000")
    parser.add_argument("--sweep-episodes", type=int, default=24)
    parser.add_argument("--audit-episodes", type=int, default=24)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--history-draws", type=int, default=8)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=4.0e-5)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--eval-episodes", type=int, default=220)
    parser.add_argument("--cont-episodes", type=int, default=5)
    parser.add_argument("--cont-weight", type=float, default=0.12)
    parser.add_argument("--cont-tail-weight", type=float, default=0.15)
    parser.add_argument("--cont-max-weight", type=float, default=0.0)
    parser.add_argument("--online-hard-weight", type=float, default=0.07)
    parser.add_argument("--online-hard-tail-weight", type=float, default=0.22)
    parser.add_argument("--online-hard-max-weight", type=float, default=0.08)
    parser.add_argument("--online-hard-refresh-every", type=int, default=1)
    parser.add_argument("--online-hard-episodes", type=int, default=10)
    parser.add_argument("--online-hard-top-k", type=int, default=4)
    parser.add_argument("--online-hard-history-draws", type=int, default=24)
    parser.add_argument("--online-hard-grid", type=int, default=2)
    parser.add_argument("--online-hard-radius", type=float, default=0.45)
    parser.add_argument("--selection-online-hard-weight", type=float, default=0.12)
    parser.add_argument("--validation-online-hard-episodes", type=int, default=8)
    parser.add_argument("--selection-objective-weight", type=float, default=1.10)
    parser.add_argument("--selection-max-weight", type=float, default=0.20)
    parser.add_argument("--selection-efficiency-target", type=float, default=0.90)
    parser.add_argument("--selection-efficiency-weight", type=float, default=5.0)
    parser.add_argument("--p95-regret-weight", type=float, default=1.5)
    parser.add_argument("--p95-regret-target", type=float, default=0.08)
    parser.add_argument("--anchor-weight", type=float, default=120.0)
    parser.add_argument("--baseline-delta-weight", type=float, default=0.0)
    parser.add_argument("--baseline-delta-tail-weight", type=float, default=1.0)
    parser.add_argument("--baseline-delta-max-weight", type=float, default=1.0)
    parser.add_argument("--baseline-delta-margin", type=float, default=0.0)
    parser.add_argument("--baseline-delta-episodes", type=int, default=0)
    parser.add_argument("--baseline-delta-seed-offset", type=int, default=2_100_000)
    parser.add_argument("--selection-baseline-delta-weight", type=float, default=0.0)
    parser.add_argument("--include-baseline-selection", action="store_true")
    parser.add_argument("--baseline-selection-margin", type=float, default=0.0)
    parser.add_argument("--fixed-validation-online-hard", action="store_true")
    parser.add_argument("--screen-mean-tolerance", type=float, default=0.0)
    parser.add_argument("--screen-p95-tolerance", type=float, default=0.0)
    parser.add_argument("--screen-max-tolerance", type=float, default=0.0)
    parser.add_argument("--screen-min-efficiency", type=float, default=0.0)
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    offsets = parse_ints(args.offsets)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    by_seed: list[dict[str, Any]] = []
    for seed in seeds:
        base_run = Path(args.base_template.format(seed=seed))
        if not (base_run / "model.pt").exists():
            raise FileNotFoundError(f"Missing model.pt for seed {seed}: {base_run}")
        out_dir = out_root / f"seed{seed}"
        repaired_run = out_dir / "repaired"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_repair and not (args.reuse_existing and (repaired_run / "model.pt").exists()):
            run_command(
                [
                    sys.executable,
                    "online_hardcase_repair.py",
                    "--base-run",
                    str(base_run),
                    "--out-root",
                    str(out_dir),
                    "--steps",
                    str(args.steps),
                    "--lr",
                    str(args.lr),
                    "--batch-size",
                    str(args.batch_size),
                    "--eval-episodes",
                    str(args.eval_episodes),
                    "--cont-episodes",
                    str(args.cont_episodes),
                    "--cont-weight",
                    str(args.cont_weight),
                    "--cont-tail-weight",
                    str(args.cont_tail_weight),
                    "--cont-max-weight",
                    str(args.cont_max_weight),
                    "--audit-episodes",
                    str(args.audit_episodes),
                    "--grid",
                    str(args.grid),
                    "--history-draws",
                    str(args.history_draws),
                    "--exit-thresholds",
                    args.exit_thresholds,
                    "--chunk-size",
                    str(args.chunk_size),
                    "--seed-offset-audit",
                    str(offsets[0]),
                    "--online-hard-weight",
                    str(args.online_hard_weight),
                    "--online-hard-tail-weight",
                    str(args.online_hard_tail_weight),
                    "--online-hard-max-weight",
                    str(args.online_hard_max_weight),
                    "--online-hard-refresh-every",
                    str(args.online_hard_refresh_every),
                    "--online-hard-episodes",
                    str(args.online_hard_episodes),
                    "--online-hard-top-k",
                    str(args.online_hard_top_k),
                    "--online-hard-history-draws",
                    str(args.online_hard_history_draws),
                    "--online-hard-grid",
                    str(args.online_hard_grid),
                    "--online-hard-radius",
                    str(args.online_hard_radius),
                    "--selection-online-hard-weight",
                    str(args.selection_online_hard_weight),
                    "--validation-online-hard-episodes",
                    str(args.validation_online_hard_episodes),
                    "--selection-objective-weight",
                    str(args.selection_objective_weight),
                    "--selection-max-weight",
                    str(args.selection_max_weight),
                    "--selection-efficiency-target",
                    str(args.selection_efficiency_target),
                    "--selection-efficiency-weight",
                    str(args.selection_efficiency_weight),
                    "--p95-regret-weight",
                    str(args.p95_regret_weight),
                    "--p95-regret-target",
                    str(args.p95_regret_target),
                    "--anchor-weight",
                    str(args.anchor_weight),
                    "--baseline-delta-weight",
                    str(args.baseline_delta_weight),
                    "--baseline-delta-tail-weight",
                    str(args.baseline_delta_tail_weight),
                    "--baseline-delta-max-weight",
                    str(args.baseline_delta_max_weight),
                    "--baseline-delta-margin",
                    str(args.baseline_delta_margin),
                    "--baseline-delta-episodes",
                    str(args.baseline_delta_episodes),
                    "--baseline-delta-seed-offset",
                    str(args.baseline_delta_seed_offset),
                    "--selection-baseline-delta-weight",
                    str(args.selection_baseline_delta_weight),
                    *(
                        ["--include-baseline-selection"]
                        if args.include_baseline_selection
                        else []
                    ),
                    "--baseline-selection-margin",
                    str(args.baseline_selection_margin),
                    *(
                        ["--fixed-validation-online-hard"]
                        if args.fixed_validation_online_hard
                        else []
                    ),
                    "--seed",
                    str(args.seed_base + seed),
                    "--skip-audit",
                ]
            )

        if not (repaired_run / "model.pt").exists():
            raise FileNotFoundError(f"Missing repaired model for seed {seed}: {repaired_run}")

        sweep_dir = out_dir / "paired_sweep"
        if not args.skip_sweep and not (args.reuse_existing and (sweep_dir / "paired_continuation_deltas.csv").exists()):
            run_command(
                [
                    sys.executable,
                    "paired_continuation_sweep.py",
                    "--runs",
                    str(base_run),
                    str(repaired_run),
                    "--seed-offsets",
                    ",".join(str(offset) for offset in offsets),
                    "--episodes",
                    str(args.sweep_episodes),
                    "--grid",
                    str(args.grid),
                    "--family",
                    "history",
                    "--history-draws",
                    str(args.history_draws),
                    "--exit-thresholds",
                    args.exit_thresholds,
                    "--chunk-size",
                    str(args.chunk_size),
                    "--out-dir",
                    str(sweep_dir),
                ]
            )

        by_seed.append(collect_seed_summary(seed, out_dir, base_run, repaired_run))

    by_seed_df = pd.DataFrame(by_seed)
    by_seed_df.to_csv(out_root / "online_hardcase_multiseed_by_seed.csv", index=False)

    all_delta_tables = []
    for seed in seeds:
        delta_path = out_root / f"seed{seed}" / "paired_sweep" / "paired_continuation_deltas.csv"
        if delta_path.exists():
            frame = pd.read_csv(delta_path)
            frame.insert(0, "seed", seed)
            all_delta_tables.append(frame)
    if all_delta_tables:
        all_deltas = pd.concat(all_delta_tables, ignore_index=True)
        all_deltas.to_csv(out_root / "online_hardcase_multiseed_deltas.csv", index=False)
        numeric_cols = [col for col in all_deltas.columns if col.startswith("delta_")]
        delta_summary = all_deltas[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})
        delta_summary.to_csv(out_root / "online_hardcase_multiseed_delta_summary.csv", index=False)
        wins = {col: int((all_deltas[col] < 0.0).sum()) for col in numeric_cols}
        adoption_by_seed = {}
        for row in by_seed:
            seed = int(row["seed"])
            repair_efficiency = row.get("repair_efficiency")
            enough_efficiency = repair_efficiency is None or float(repair_efficiency) >= args.screen_min_efficiency
            adopt = (
                enough_efficiency
                and float(row.get("mean_delta_continuation_mean_regret") or 0.0) <= args.screen_mean_tolerance
                and float(row.get("mean_delta_continuation_p95_regret") or 0.0) <= args.screen_p95_tolerance
                and float(row.get("mean_delta_continuation_max_regret") or 0.0) <= args.screen_max_tolerance
            )
            adoption_by_seed[seed] = adopt
            row["screened_adopted"] = adopt
        screened_deltas = all_deltas.copy()
        for column in numeric_cols:
            screened_deltas[column] = screened_deltas.apply(
                lambda item: item[column] if adoption_by_seed.get(int(item["seed"]), False) else 0.0,
                axis=1,
            )
        screened_deltas.to_csv(out_root / "online_hardcase_screened_deltas.csv", index=False)
        screened_summary = screened_deltas[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})
        screened_summary.to_csv(out_root / "online_hardcase_screened_delta_summary.csv", index=False)
        screened_wins = {col: int((screened_deltas[col] < 0.0).sum()) for col in numeric_cols}
    else:
        delta_summary = pd.DataFrame()
        screened_summary = pd.DataFrame()
        wins = {}
        screened_wins = {}
        adoption_by_seed = {int(row["seed"]): False for row in by_seed}
    pd.DataFrame(by_seed).to_csv(out_root / "online_hardcase_multiseed_by_seed.csv", index=False)

    summary: dict[str, Any] = {
        "seeds": seeds,
        "offsets": offsets,
        "args": vars(args),
        "seed_count": len(seeds),
        "seed_rows": by_seed,
        "negative_delta_wins": wins,
        "screening_rule": {
            "mean_tolerance": args.screen_mean_tolerance,
            "p95_tolerance": args.screen_p95_tolerance,
            "max_tolerance": args.screen_max_tolerance,
            "min_efficiency": args.screen_min_efficiency,
        },
        "screened_adoption_by_seed": adoption_by_seed,
        "screened_negative_delta_wins": screened_wins,
        "screened_delta_summary": screened_summary.to_dict(orient="records") if not screened_summary.empty else [],
        "delta_summary": delta_summary.to_dict(orient="records") if not delta_summary.empty else [],
    }
    (out_root / "online_hardcase_multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
