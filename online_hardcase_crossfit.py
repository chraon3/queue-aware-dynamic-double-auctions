from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


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


def summarize_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [column for column in frame.columns if column.startswith("delta_")]
    if not numeric_cols:
        return pd.DataFrame()
    return frame[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})


def wins(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int((frame[column] < 0.0).sum()) for column in frame.columns if column.startswith("delta_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="81,83,89,91,93,97,101,103,107,109")
    parser.add_argument("--screen-root", default="experiments/online_hardcase_multiseed_screen")
    parser.add_argument("--out-dir", default="experiments/online_hardcase_crossfit")
    parser.add_argument("--base-template", default="experiments/dynamic_patience_value_seed{seed}")
    parser.add_argument("--candidate-template", default="{screen_root}/seed{seed}/repaired")
    parser.add_argument("--final-offsets", default="210000,230000")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--family", choices=["age", "state", "history"], default="history")
    parser.add_argument("--history-draws", type=int, default=8)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    final_offsets = parse_ints(args.final_offsets)
    screen_root = Path(args.screen_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    screen_summary = read_json(screen_root / "online_hardcase_multiseed_summary.json")
    adoption_raw = screen_summary.get("screened_adoption_by_seed", {})
    if not adoption_raw:
        raise FileNotFoundError(f"Missing screened adoption decisions under {screen_root}")
    adoption = {int(seed): bool(value) for seed, value in adoption_raw.items()}

    all_deltas = []
    by_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        base_run = Path(args.base_template.format(seed=seed, screen_root=str(screen_root)))
        candidate_run = Path(args.candidate_template.format(seed=seed, screen_root=str(screen_root)))
        if not (base_run / "model.pt").exists():
            raise FileNotFoundError(f"Missing base model for seed {seed}: {base_run}")
        if not (candidate_run / "model.pt").exists():
            raise FileNotFoundError(f"Missing candidate model for seed {seed}: {candidate_run}")

        seed_dir = out_dir / f"seed{seed}"
        sweep_dir = seed_dir / "final_sweep"
        if not args.reuse_existing or not (sweep_dir / "paired_continuation_deltas.csv").exists():
            run_command(
                [
                    sys.executable,
                    "paired_continuation_sweep.py",
                    "--runs",
                    str(base_run),
                    str(candidate_run),
                    "--seed-offsets",
                    ",".join(str(offset) for offset in final_offsets),
                    "--episodes",
                    str(args.episodes),
                    "--grid",
                    str(args.grid),
                    "--family",
                    args.family,
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
        delta_path = sweep_dir / "paired_continuation_deltas.csv"
        if not delta_path.exists():
            raise FileNotFoundError(f"Missing final paired deltas for seed {seed}: {delta_path}")
        frame = pd.read_csv(delta_path)
        frame.insert(0, "seed", seed)
        frame["screened_adopted"] = adoption.get(seed, False)
        all_deltas.append(frame)
        numeric_cols = [column for column in frame.columns if column.startswith("delta_")]
        seed_row: dict[str, Any] = {
            "seed": seed,
            "screened_adopted": adoption.get(seed, False),
            "offset_count": int(len(frame)),
            "base_run": str(base_run),
            "candidate_run": str(candidate_run),
        }
        for column in numeric_cols:
            seed_row[f"mean_{column}"] = float(frame[column].mean())
        by_seed_rows.append(seed_row)

    final_deltas = pd.concat(all_deltas, ignore_index=True)
    final_deltas.to_csv(out_dir / "crossfit_final_unfiltered_deltas.csv", index=False)
    final_summary = summarize_deltas(final_deltas)
    final_summary.to_csv(out_dir / "crossfit_final_unfiltered_summary.csv", index=False)

    numeric_cols = [column for column in final_deltas.columns if column.startswith("delta_")]
    screened_final = final_deltas.copy()
    for column in numeric_cols:
        screened_final[column] = screened_final.apply(
            lambda row: row[column] if bool(row["screened_adopted"]) else 0.0,
            axis=1,
        )
    screened_final.to_csv(out_dir / "crossfit_final_screened_deltas.csv", index=False)
    screened_summary = summarize_deltas(screened_final)
    screened_summary.to_csv(out_dir / "crossfit_final_screened_summary.csv", index=False)

    by_seed = pd.DataFrame(by_seed_rows)
    by_seed.to_csv(out_dir / "crossfit_by_seed.csv", index=False)
    summary = {
        "seeds": seeds,
        "selection_root": str(screen_root),
        "selection_offsets": screen_summary.get("offsets", []),
        "final_offsets": final_offsets,
        "adoption_by_seed": adoption,
        "adopted_count": int(sum(adoption.get(seed, False) for seed in seeds)),
        "args": vars(args),
        "unfiltered_negative_delta_wins": wins(final_deltas),
        "screened_negative_delta_wins": wins(screened_final),
        "unfiltered_summary": final_summary.to_dict(orient="records"),
        "screened_summary": screened_summary.to_dict(orient="records"),
    }
    (out_dir / "online_hardcase_crossfit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
