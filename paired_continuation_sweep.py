from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dynamic_continuation_audit import audit_run, parse_exit_thresholds


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


def parse_offsets(text: str) -> list[int]:
    offsets = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not offsets:
        raise ValueError("At least one seed offset is required.")
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--seed-offsets", default="170000,180000,190000,200000")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--family", choices=["age", "state", "history", "history_nonlinear", "history_piecewise"], default="history")
    parser.add_argument("--history-draws", type=int, default=8)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--out-dir", default="experiments/paired_continuation_sweep")
    args = parser.parse_args()

    run_dirs = [Path(path) for path in args.runs]
    for run_dir in run_dirs:
        if not (run_dir / "model.pt").exists():
            raise FileNotFoundError(f"Missing model.pt under {run_dir}")
    offsets = parse_offsets(args.seed_offsets)
    exit_thresholds = parse_exit_thresholds(args.exit_thresholds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for offset in offsets:
        print(f"Paired continuation sweep offset={offset}", flush=True)
        for run_dir in run_dirs:
            row, _details = audit_run(
                run_dir,
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
            row["paired_offset"] = float(offset)
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "paired_continuation_by_run.csv", index=False)
    summary: dict[str, object] = {"runs": [str(path) for path in run_dirs], "offsets": offsets}
    if len(run_dirs) == 2:
        left_name = run_dirs[0].name
        right_name = run_dirs[1].name
        deltas = []
        for offset in offsets:
            offset_rows = table[table["paired_offset"] == float(offset)].set_index("run")
            if left_name not in offset_rows.index or right_name not in offset_rows.index:
                continue
            row: dict[str, float | str] = {"paired_offset": float(offset), "baseline_run": left_name, "candidate_run": right_name}
            for column in DELTA_COLUMNS:
                row[f"delta_{column}"] = float(offset_rows.loc[right_name, column]) - float(offset_rows.loc[left_name, column])
            deltas.append(row)
        delta_table = pd.DataFrame(deltas)
        delta_table.to_csv(out_dir / "paired_continuation_deltas.csv", index=False)
        numeric = delta_table.drop(columns=["baseline_run", "candidate_run"]) if not delta_table.empty else delta_table
        delta_summary = numeric.agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"}) if not numeric.empty else pd.DataFrame()
        if not delta_summary.empty:
            delta_summary.to_csv(out_dir / "paired_continuation_delta_summary.csv", index=False)
        wins = {}
        for column in DELTA_COLUMNS:
            delta_column = f"delta_{column}"
            if delta_column in delta_table:
                wins[column] = int((delta_table[delta_column] < 0.0).sum())
        summary.update(
            {
                "baseline_run": left_name,
                "candidate_run": right_name,
                "offset_count": len(deltas),
                "negative_delta_wins": wins,
            }
        )
        if not delta_summary.empty:
            summary["delta_summary"] = delta_summary.to_dict(orient="records")

    (out_dir / "paired_continuation_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
