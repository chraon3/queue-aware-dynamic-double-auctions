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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", default="experiments/dynamic_patience_value_seed81")
    parser.add_argument("--out-root", default="experiments/hardcase_mining_repair_seed81")
    parser.add_argument("--mine-episodes", type=int, default=40)
    parser.add_argument("--validation-mine-episodes", type=int, default=40)
    parser.add_argument("--audit-episodes", type=int, default=60)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--history-draws", type=int, default=12)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--top-k-details", type=int, default=10)
    parser.add_argument("--hard-case-top-k", type=int, default=12)
    parser.add_argument("--seed-offset-train", type=int, default=130_000)
    parser.add_argument("--seed-offset-validation", type=int, default=150_000)
    parser.add_argument("--seed-offset-audit", type=int, default=170_000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=360)
    parser.add_argument("--cont-episodes", type=int, default=10)
    parser.add_argument("--cont-weight", type=float, default=0.30)
    parser.add_argument("--cont-tail-weight", type=float, default=0.20)
    parser.add_argument("--cont-max-weight", type=float, default=0.10)
    parser.add_argument("--hard-case-weight", type=float, default=0.22)
    parser.add_argument("--hard-case-tail-weight", type=float, default=0.30)
    parser.add_argument("--hard-case-max-weight", type=float, default=0.30)
    parser.add_argument("--selection-hard-case-weight", type=float, default=0.45)
    parser.add_argument("--selection-objective-weight", type=float, default=0.90)
    parser.add_argument("--selection-max-weight", type=float, default=0.45)
    parser.add_argument("--selection-efficiency-target", type=float, default=0.90)
    parser.add_argument("--selection-efficiency-weight", type=float, default=4.0)
    parser.add_argument("--p95-regret-weight", type=float, default=1.5)
    parser.add_argument("--p95-regret-target", type=float, default=0.08)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=910_081)
    parser.add_argument("--skip-mine", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    train_mine_dir = out_root / "mine_train"
    validation_mine_dir = out_root / "mine_validation"
    repair_dir = out_root / "repaired"
    audit_dir = out_root / "paired_audit"
    out_root.mkdir(parents=True, exist_ok=True)

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
        str(args.top_k_details),
    ]

    if not args.skip_mine:
        run_command(
            [
                sys.executable,
                "dynamic_continuation_audit.py",
                "--runs",
                args.base_run,
                "--episodes",
                str(args.mine_episodes),
                "--seed-offset",
                str(args.seed_offset_train),
                "--out-dir",
                str(train_mine_dir),
                *common_audit_args,
            ]
        )
        run_command(
            [
                sys.executable,
                "dynamic_continuation_audit.py",
                "--runs",
                args.base_run,
                "--episodes",
                str(args.validation_mine_episodes),
                "--seed-offset",
                str(args.seed_offset_validation),
                "--out-dir",
                str(validation_mine_dir),
                *common_audit_args,
            ]
        )

    train_cases = train_mine_dir / "continuation_audit_worst_cases.csv"
    validation_cases = validation_mine_dir / "continuation_audit_worst_cases.csv"
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
                "--hard-case-csv",
                str(train_cases),
                "--validation-hard-case-csv",
                str(validation_cases),
                "--hard-case-top-k",
                str(args.hard_case_top_k),
                "--hard-case-weight",
                str(args.hard_case_weight),
                "--hard-case-tail-weight",
                str(args.hard_case_tail_weight),
                "--hard-case-max-weight",
                str(args.hard_case_max_weight),
                "--selection-hard-case-weight",
                str(args.selection_hard_case_weight),
                "--anchor-weight",
                str(args.anchor_weight),
                "--tail-alpha",
                "0.75",
                "--p95-regret-weight",
                str(args.p95_regret_weight),
                "--p95-regret-target",
                str(args.p95_regret_target),
                "--select-best",
                "--validation-every",
                "2",
                "--validation-episodes",
                "12",
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
        "train_mine_dir": str(train_mine_dir),
        "validation_mine_dir": str(validation_mine_dir),
        "audit_dir": str(audit_dir),
        "args": vars(args),
        "train_hard_cases": read_csv_rows(train_cases)[: min(args.hard_case_top_k or args.top_k_details * 2, 20)],
        "validation_hard_cases": read_csv_rows(validation_cases)[: min(args.hard_case_top_k or args.top_k_details * 2, 20)],
    }
    repair_metrics = repair_dir / "metrics.json"
    if repair_metrics.exists():
        summary["repair_metrics"] = json.loads(repair_metrics.read_text(encoding="utf-8"))
    audit_table = audit_dir / "continuation_audit_by_run.csv"
    if audit_table.exists():
        summary["paired_audit_rows"] = read_csv_rows(audit_table)
    (out_root / "hardcase_mining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
