from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dynamic_continuation_audit import audit_run, parse_exit_thresholds


METRIC_COLUMNS = [
    "continuation_mean_regret",
    "continuation_p95_regret",
    "continuation_max_regret",
    "no_exit_continuation_mean_regret",
    "no_exit_continuation_p95_regret",
]


def parse_offsets(text: str) -> list[int]:
    offsets = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not offsets:
        raise ValueError("At least one audit offset is required.")
    return offsets


def collect_run_dirs(patterns: list[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for pattern in patterns:
        run_dirs.extend(sorted(Path().glob(pattern)))
    run_dirs = [path for path in run_dirs if (path / "model.pt").exists()]
    if not run_dirs:
        raise FileNotFoundError("No run directories with model.pt were found.")
    return run_dirs


def write_offset_table(offset_summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{rrrrr}", "\\toprule"]
    rows.append("Offset & Mean & P95 & Max & No-exit mean \\\\")
    rows.append("\\midrule")
    for _, row in offset_summary.iterrows():
        rows.append(
            " & ".join(
                [
                    f"{int(row['audit_offset'])}",
                    f"{row['continuation_mean_regret']:.3f}",
                    f"{row['continuation_p95_regret']:.3f}",
                    f"{row['continuation_max_regret']:.3f}",
                    f"{row['no_exit_continuation_mean_regret']:.3f}",
                ]
            )
            + " \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--offsets", default="80000,120000,160000,200000")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--family", choices=["age", "state", "history", "history_nonlinear", "history_piecewise"], default="history")
    parser.add_argument("--history-draws", type=int, default=24)
    parser.add_argument("--exit-thresholds", default="2,4,99")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--seed-stride", type=int, default=997)
    parser.add_argument("--out-dir", default="experiments/dynamic_continuation_audit_multi_offset")
    parser.add_argument("--paper-table", default="paper/tables/continuation_multi_offset.tex")
    parser.add_argument("--save-samples", action="store_true")
    args = parser.parse_args()

    run_dirs = collect_run_dirs(args.runs)
    offsets = parse_offsets(args.offsets)
    exit_thresholds = parse_exit_thresholds(args.exit_thresholds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    sample_paths: list[Path] = []
    for offset in offsets:
        print(f"Continuation audit offset={offset}", flush=True)
        for idx, run_dir in enumerate(run_dirs):
            seed_offset = offset + idx * args.seed_stride
            sample_path = None
            if args.save_samples:
                sample_path = out_dir / f"{run_dir.name}_offset{offset}_continuation_samples.csv"
            row, _details = audit_run(
                run_dir,
                args.episodes,
                args.grid,
                args.radius,
                args.family,
                args.history_draws,
                exit_thresholds,
                args.chunk_size,
                seed_offset,
                0,
                sample_path,
            )
            row["audit_offset"] = float(offset)
            rows.append(row)
            if sample_path is not None:
                sample_paths.append(sample_path)

    by_run = pd.DataFrame(rows)
    by_run.to_csv(out_dir / "continuation_multi_offset_by_run.csv", index=False)
    if args.save_samples:
        sample_frames = [pd.read_csv(path) for path in sample_paths if path.exists()]
        samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
        samples.to_csv(out_dir / "continuation_multi_offset_samples.csv", index=False)

    offset_summary = by_run.groupby("audit_offset", as_index=False)[METRIC_COLUMNS].mean()
    offset_summary.to_csv(out_dir / "continuation_multi_offset_by_offset.csv", index=False)
    overall = offset_summary[METRIC_COLUMNS].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})
    overall.to_csv(out_dir / "continuation_multi_offset_summary.csv", index=False)
    paper_table = Path(args.paper_table)
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    write_offset_table(offset_summary, paper_table)
    print(offset_summary.to_string(index=False), flush=True)
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
