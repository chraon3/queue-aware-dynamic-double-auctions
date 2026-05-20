from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", default="experiments/dynamic_patience_value_seed81")
    parser.add_argument("--out-root", default="experiments/online_hardcase_repair_seed81")
    parser.add_argument("--audit-episodes", type=int, default=60)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--history-draws", type=int, default=12)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--seed-offset-audit", type=int, default=190_000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=360)
    parser.add_argument("--cont-episodes", type=int, default=8)
    parser.add_argument("--cont-weight", type=float, default=0.12)
    parser.add_argument("--cont-tail-weight", type=float, default=0.15)
    parser.add_argument("--cont-max-weight", type=float, default=0.0)
    parser.add_argument("--online-hard-weight", type=float, default=0.08)
    parser.add_argument("--online-hard-tail-weight", type=float, default=0.25)
    parser.add_argument("--online-hard-max-weight", type=float, default=0.10)
    parser.add_argument("--online-hard-refresh-every", type=int, default=2)
    parser.add_argument("--online-hard-episodes", type=int, default=20)
    parser.add_argument("--online-hard-top-k", type=int, default=5)
    parser.add_argument("--online-hard-history-draws", type=int, default=32)
    parser.add_argument("--online-hard-grid", type=int, default=2)
    parser.add_argument("--online-hard-radius", type=float, default=0.45)
    parser.add_argument("--selection-online-hard-weight", type=float, default=0.15)
    parser.add_argument("--validation-online-hard-episodes", type=int, default=12)
    parser.add_argument("--selection-objective-weight", type=float, default=1.10)
    parser.add_argument("--selection-max-weight", type=float, default=0.20)
    parser.add_argument("--selection-efficiency-target", type=float, default=0.90)
    parser.add_argument("--selection-efficiency-weight", type=float, default=5.0)
    parser.add_argument("--p95-regret-weight", type=float, default=1.5)
    parser.add_argument("--p95-regret-target", type=float, default=0.08)
    parser.add_argument("--anchor-weight", type=float, default=100.0)
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
    parser.add_argument("--seed", type=int, default=930_081)
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    repair_dir = out_root / "repaired"
    audit_dir = out_root / "paired_audit"
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_repair:
        run_command(
            [
                sys.executable,
                "finetune_continuation_regret.py",
                "--run-dir",
                args.base_run,
                "--out-dir",
                str(repair_dir),
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
                "--tail-alpha",
                "0.75",
                "--p95-regret-weight",
                str(args.p95_regret_weight),
                "--p95-regret-target",
                str(args.p95_regret_target),
                "--select-best",
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
                "--validation-every",
                "2",
                "--validation-episodes",
                "10",
                "--validation-objective-episodes",
                "160",
                "--selection-objective-weight",
                str(args.selection_objective_weight),
                "--selection-max-weight",
                str(args.selection_max_weight),
                "--selection-efficiency-target",
                str(args.selection_efficiency_target),
                "--selection-efficiency-weight",
                str(args.selection_efficiency_weight),
                "--grid",
                str(args.grid),
                "--history-draws",
                str(args.history_draws),
                "--exit-thresholds",
                args.exit_thresholds,
                "--chunk-size",
                str(args.chunk_size),
                "--seed",
                str(args.seed),
            ]
        )

    common_audit_args = [
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
        "--save-details",
        "--top-k-details",
        "8",
    ]
    if not args.skip_audit:
        run_command(
            [
                sys.executable,
                "dynamic_continuation_audit.py",
                "--runs",
                args.base_run,
                str(repair_dir),
                "--episodes",
                str(args.audit_episodes),
                "--seed-offset",
                str(args.seed_offset_audit),
                "--paired-draws",
                "--out-dir",
                str(audit_dir),
                *common_audit_args,
            ]
        )

    summary: dict[str, object] = {
        "base_run": args.base_run,
        "repair_dir": str(repair_dir),
        "audit_dir": str(audit_dir),
        "args": vars(args),
        "repair_metrics": read_json(repair_dir / "metrics.json"),
        "finetune_manifest": read_json(repair_dir / "finetune_manifest.json"),
        "online_hard_history": read_json(repair_dir / "online_hard_history.json"),
        "paired_audit_rows": read_csv_rows(audit_dir / "continuation_audit_by_run.csv"),
    }
    (out_root / "online_hardcase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
