from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DELTA_COLUMNS = [
    "delta_buyer_continuation_mean_regret",
    "delta_seller_continuation_mean_regret",
    "delta_continuation_mean_regret",
    "delta_continuation_p95_regret",
    "delta_continuation_max_regret",
    "delta_no_exit_continuation_mean_regret",
    "delta_no_exit_continuation_p95_regret",
    "delta_no_exit_continuation_max_regret",
]


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def alpha_name(alpha: float) -> str:
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return "alpha_" + text.replace("-", "m").replace(".", "p")


def run_command(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [column for column in DELTA_COLUMNS if column in frame.columns]
    if not cols:
        return pd.DataFrame()
    return frame[cols].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})


def wins(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int((frame[column] < 0.0).sum()) for column in DELTA_COLUMNS if column in frame.columns}


def best_screened_row(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing interpolation summary: {summary_path}")
    frame = pd.read_csv(summary_path)
    if frame.empty:
        raise ValueError(f"Interpolation summary is empty: {summary_path}")
    screened = frame[frame["screened_adopted"].astype(str).str.lower().isin(["true", "1"])]
    if screened.empty:
        fallback = frame[frame["alpha"].astype(float) == 0.0]
        if fallback.empty:
            raise ValueError(f"No screened alpha and no alpha=0 fallback in {summary_path}")
        return fallback.iloc[0].to_dict()
    return screened.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="81,83,89,91,93,97,101,103,107,109")
    parser.add_argument("--base-template", default="experiments/dynamic_patience_value_seed{seed}")
    parser.add_argument("--repair-root", default="experiments/online_hardcase_baselinedelta_multiseed_guard32")
    parser.add_argument("--selection-root", default="experiments/risk_controlled_interpolation_multiseed")
    parser.add_argument("--out-dir", default="experiments/risk_controlled_interpolation_crossfit")
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.35,0.4,0.5,0.75,1.0")
    parser.add_argument("--selection-offsets", default="170000,190000")
    parser.add_argument("--calibration-offsets", default="")
    parser.add_argument("--final-offsets", default="210000,230000")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=220)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--history-draws", type=int, default=8)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--min-efficiency", type=float, default=0.85)
    parser.add_argument("--effective-update-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--calibration-tolerance", type=float, default=0.0)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    selection_offsets = parse_ints(args.selection_offsets)
    calibration_offsets = parse_ints(args.calibration_offsets) if args.calibration_offsets.strip() else []
    final_offsets = parse_ints(args.final_offsets)
    alphas = parse_floats(args.alphas)
    repair_root = Path(args.repair_root)
    selection_root = Path(args.selection_root)
    out_dir = Path(args.out_dir)
    repair_root.mkdir(parents=True, exist_ok=True)
    selection_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_repair:
        repair_done = all((repair_root / f"seed{seed}" / "repaired" / "model.pt").exists() for seed in seeds)
        if not (args.reuse_existing and repair_done):
            run_command(
                [
                    sys.executable,
                    "online_hardcase_multiseed.py",
                    "--seeds",
                    ",".join(str(seed) for seed in seeds),
                    "--out-root",
                    str(repair_root),
                    "--steps",
                    "3",
                    "--lr",
                    "0.00004",
                    "--batch-size",
                    "72",
                    "--eval-episodes",
                    str(args.eval_episodes),
                    "--cont-episodes",
                    "5",
                    "--audit-episodes",
                    "16",
                    "--offsets",
                    ",".join(str(offset) for offset in selection_offsets),
                    "--sweep-episodes",
                    str(args.episodes),
                    "--history-draws",
                    str(args.history_draws),
                    "--online-hard-episodes",
                    "10",
                    "--online-hard-top-k",
                    "4",
                    "--online-hard-history-draws",
                    "24",
                    "--online-hard-refresh-every",
                    "1",
                    "--online-hard-weight",
                    "0.06",
                    "--online-hard-tail-weight",
                    "0.18",
                    "--online-hard-max-weight",
                    "0.06",
                    "--baseline-delta-weight",
                    "2.5",
                    "--baseline-delta-tail-weight",
                    "2.0",
                    "--baseline-delta-max-weight",
                    "3.0",
                    "--baseline-delta-episodes",
                    "32",
                    "--selection-baseline-delta-weight",
                    "3.0",
                    "--selection-online-hard-weight",
                    "0.10",
                    "--validation-online-hard-episodes",
                    "8",
                    "--selection-objective-weight",
                    "1.05",
                    "--selection-max-weight",
                    "0.20",
                    "--selection-efficiency-target",
                    "0.90",
                    "--selection-efficiency-weight",
                    "5.0",
                    "--anchor-weight",
                    "120.0",
                    "--include-baseline-selection",
                    "--fixed-validation-online-hard",
                    "--screen-min-efficiency",
                    str(args.min_efficiency),
                    *(
                        ["--reuse-existing"]
                        if args.reuse_existing
                        else []
                    ),
                ]
            )

    selection_rows: list[dict[str, Any]] = []
    calibration_delta_tables: list[pd.DataFrame] = []
    for seed in seeds:
        base_run = Path(args.base_template.format(seed=seed))
        candidate_run = repair_root / f"seed{seed}" / "repaired"
        if not (base_run / "model.pt").exists():
            raise FileNotFoundError(f"Missing base model for seed {seed}: {base_run}")
        if not (candidate_run / "model.pt").exists():
            raise FileNotFoundError(f"Missing guarded repair for seed {seed}: {candidate_run}")
        seed_selection_root = selection_root / f"seed{seed}"
        summary_path = seed_selection_root / "interpolation_summary.csv"
        if not args.skip_selection and not (args.reuse_existing and summary_path.exists()):
            run_command(
                [
                    sys.executable,
                    "risk_controlled_interpolation_sweep.py",
                    "--base-run",
                    str(base_run),
                    "--candidate-run",
                    str(candidate_run),
                    "--out-root",
                    str(seed_selection_root),
                    "--alphas",
                    ",".join(str(alpha) for alpha in alphas),
                    "--seed-offsets",
                    ",".join(str(offset) for offset in selection_offsets),
                    "--episodes",
                    str(args.episodes),
                    "--eval-episodes",
                    str(args.eval_episodes),
                    "--grid",
                    str(args.grid),
                    "--history-draws",
                    str(args.history_draws),
                    "--exit-thresholds",
                    args.exit_thresholds,
                    "--chunk-size",
                    str(args.chunk_size),
                    "--min-efficiency",
                    str(args.min_efficiency),
                ]
            )
        best = best_screened_row(summary_path)
        pre_veto_alpha = float(best["alpha"])
        selected_alpha = pre_veto_alpha
        selected_run = seed_selection_root / alpha_name(selected_alpha)
        calibration_vetoed = False
        calibration_positive_max = 0.0
        calibration_metrics: dict[str, float] = {}
        if calibration_offsets and not args.skip_calibration:
            seed_calibration_dir = out_dir / f"seed{seed}" / f"calibration_sweep_{alpha_name(selected_alpha)}"
            calibration_delta_path = seed_calibration_dir / "paired_continuation_deltas.csv"
            if not (args.reuse_existing and calibration_delta_path.exists()):
                run_command(
                    [
                        sys.executable,
                        "paired_continuation_sweep.py",
                        "--runs",
                        str(base_run),
                        str(selected_run),
                        "--seed-offsets",
                        ",".join(str(offset) for offset in calibration_offsets),
                        "--episodes",
                        str(args.episodes),
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
                        str(seed_calibration_dir),
                    ]
                )
            calibration_frame = pd.read_csv(calibration_delta_path)
            calibration_frame.insert(0, "seed", seed)
            calibration_frame["pre_veto_alpha"] = pre_veto_alpha
            calibration_delta_tables.append(calibration_frame)
            calibration_columns = [
                "delta_continuation_mean_regret",
                "delta_continuation_p95_regret",
                "delta_continuation_max_regret",
                "delta_no_exit_continuation_mean_regret",
                "delta_no_exit_continuation_p95_regret",
                "delta_no_exit_continuation_max_regret",
            ]
            for column in calibration_columns:
                if column in calibration_frame.columns:
                    calibration_metrics[f"calibration_mean_{column}"] = float(calibration_frame[column].mean())
                    calibration_metrics[f"calibration_max_{column}"] = float(calibration_frame[column].max())
            calibration_positive_max = max(
                [0.0]
                + [
                    float(calibration_frame[column].max())
                    for column in calibration_columns
                    if column in calibration_frame.columns
                ]
            )
            calibration_vetoed = selected_alpha != 0.0 and calibration_positive_max > args.calibration_tolerance
            if calibration_vetoed:
                selected_alpha = 0.0
                fallback_run = seed_selection_root / alpha_name(0.0)
                selected_run = fallback_run if fallback_run.exists() else base_run
        effective_columns = [
            "mean_delta_continuation_mean_regret",
            "mean_delta_continuation_p95_regret",
            "mean_delta_continuation_max_regret",
            "max_delta_continuation_mean_regret",
            "max_delta_continuation_p95_regret",
            "max_delta_continuation_max_regret",
        ]
        effective_update = any(
            abs(float(best.get(column, 0.0))) > args.effective_update_tolerance
            for column in effective_columns
        ) and selected_alpha != 0.0
        row = {
            "seed": seed,
            "base_run": str(base_run),
            "guarded_run": str(candidate_run),
            "selected_run": str(selected_run),
            "pre_veto_alpha": pre_veto_alpha,
            "selected_alpha": selected_alpha,
            "selected_nonbase": selected_alpha != 0.0,
            "effective_update": effective_update,
            "calibration_vetoed": calibration_vetoed,
            "calibration_positive_max": calibration_positive_max,
        }
        row.update(calibration_metrics)
        for key, value in best.items():
            if key not in row:
                row[f"selection_{key}"] = value
        selection_rows.append(row)

    selection_table = pd.DataFrame(selection_rows)
    selection_table.to_csv(out_dir / "risk_controlled_selection_by_seed.csv", index=False)
    if calibration_delta_tables:
        calibration_deltas = pd.concat(calibration_delta_tables, ignore_index=True)
        calibration_deltas.to_csv(out_dir / "risk_controlled_calibration_deltas.csv", index=False)
        calibration_summary = summarize(calibration_deltas)
        calibration_summary.to_csv(out_dir / "risk_controlled_calibration_summary.csv", index=False)
    else:
        calibration_deltas = pd.DataFrame()
        calibration_summary = pd.DataFrame()

    final_delta_tables: list[pd.DataFrame] = []
    if not args.skip_final:
        for row in selection_rows:
            seed = int(row["seed"])
            seed_dir = out_dir / f"seed{seed}" / "final_sweep"
            delta_path = seed_dir / "paired_continuation_deltas.csv"
            if not (args.reuse_existing and delta_path.exists()):
                run_command(
                    [
                        sys.executable,
                        "paired_continuation_sweep.py",
                        "--runs",
                        row["base_run"],
                        row["selected_run"],
                        "--seed-offsets",
                        ",".join(str(offset) for offset in final_offsets),
                        "--episodes",
                        str(args.episodes),
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
                        str(seed_dir),
                    ]
                )
            frame = pd.read_csv(delta_path)
            frame.insert(0, "seed", seed)
            frame["selected_alpha"] = float(row["selected_alpha"])
            frame["selected_nonbase"] = bool(row["selected_nonbase"])
            final_delta_tables.append(frame)

    if final_delta_tables:
        final_deltas = pd.concat(final_delta_tables, ignore_index=True)
        final_deltas.to_csv(out_dir / "risk_controlled_final_deltas.csv", index=False)
        final_summary = summarize(final_deltas)
        final_summary.to_csv(out_dir / "risk_controlled_final_summary.csv", index=False)
    else:
        final_deltas = pd.DataFrame()
        final_summary = pd.DataFrame()

    summary = {
        "seeds": seeds,
        "selection_offsets": selection_offsets,
        "calibration_offsets": calibration_offsets,
        "final_offsets": final_offsets,
        "alphas": alphas,
        "selected_alpha_by_seed": {str(row["seed"]): row["selected_alpha"] for row in selection_rows},
        "pre_veto_alpha_by_seed": {str(row["seed"]): row["pre_veto_alpha"] for row in selection_rows},
        "selected_nonbase_count": int(sum(bool(row["selected_nonbase"]) for row in selection_rows)),
        "effective_update_count": int(sum(bool(row["effective_update"]) for row in selection_rows)),
        "calibration_veto_count": int(sum(bool(row["calibration_vetoed"]) for row in selection_rows)),
        "selection_by_seed": selection_rows,
        "calibration_negative_delta_wins": wins(calibration_deltas) if not calibration_deltas.empty else {},
        "calibration_summary": calibration_summary.to_dict(orient="records") if not calibration_summary.empty else [],
        "final_negative_delta_wins": wins(final_deltas) if not final_deltas.empty else {},
        "final_summary": final_summary.to_dict(orient="records") if not final_summary.empty else [],
        "args": vars(args),
    }
    (out_dir / "risk_controlled_multiseed_crossfit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
