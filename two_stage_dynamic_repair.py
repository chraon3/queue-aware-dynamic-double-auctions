from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_metrics(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="experiments/two_stage_state_repair_seed81")
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--mechanism", choices=["base", "state"], default="state")
    parser.add_argument("--stage1-steps", type=int, default=180)
    parser.add_argument("--stage2-steps", type=int, default=6)
    parser.add_argument("--stage2-lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--regret-grid", type=int, default=5)
    parser.add_argument("--congestion-aux-weight", type=float, default=0.20)
    parser.add_argument("--congestion-volume-weight", type=float, default=0.80)
    parser.add_argument("--imitation-aux-weight", type=float, default=0.0)
    parser.add_argument("--cont-episodes", type=int, default=8)
    parser.add_argument("--cont-weight", type=float, default=0.18)
    parser.add_argument("--cont-tail-weight", type=float, default=0.20)
    parser.add_argument("--cont-max-weight", type=float, default=0.0)
    parser.add_argument("--history-draws", type=int, default=8)
    parser.add_argument("--exit-thresholds", default="2,3,99")
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--selection-efficiency-target", type=float, default=0.78)
    parser.add_argument("--selection-efficiency-weight", type=float, default=2.0)
    parser.add_argument("--selection-objective-weight", type=float, default=0.80)
    parser.add_argument("--audit-episodes", type=int, default=40)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    stage1_dir = out_root / "stage1_congestion_margin"
    stage2_dir = out_root / "stage2_continuation_repair"
    audit_dir = out_root / "continuation_audit"
    out_root.mkdir(parents=True, exist_ok=True)

    common_market = [
        "--max-buyers",
        "4",
        "--max-sellers",
        "4",
        "--horizon",
        "6",
        "--wait-cost",
        "0.02",
        "--arrival-prob-buyer",
        "0.75",
        "--arrival-prob-seller",
        "0.75",
        "--max-patience",
        "5",
        "--abandon-base",
        "0.01",
        "--abandon-slope",
        "0.08",
        "--feature-mode",
        "ranked",
        "--seed",
        str(args.seed),
    ]

    if not args.skip_stage1:
        run_command(
            [
                sys.executable,
                "dynamic_double_auction.py",
                "--mechanism",
                args.mechanism,
                "--batch-size",
                str(args.batch_size),
                "--train-steps",
                str(args.stage1_steps),
                "--eval-episodes",
                str(args.eval_episodes),
                "--regret-grid",
                str(args.regret_grid),
                "--regret-weight",
                "1.2" if args.mechanism == "state" else "2.0",
                "--regret-target",
                "0.04" if args.mechanism == "state" else "0.025",
                "--augmented-rho",
                "3.0" if args.mechanism == "state" else "4.0",
                "--congestion-aux-weight",
                str(args.congestion_aux_weight),
                "--congestion-volume-weight",
                str(args.congestion_volume_weight),
                "--imitation-aux-weight",
                str(args.imitation_aux_weight),
                "--select-best",
                "--selection-regret-penalty",
                "3.0",
                "--selection-min-step",
                "40",
                "--out-dir",
                str(stage1_dir),
                *common_market,
            ]
        )

    if not args.skip_stage2:
        run_command(
            [
                sys.executable,
                "finetune_continuation_regret.py",
                "--run-dir",
                str(stage1_dir),
                "--out-dir",
                str(stage2_dir),
                "--steps",
                str(args.stage2_steps),
                "--lr",
                str(args.stage2_lr),
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
                "--tail-alpha",
                "0.75",
                "--p95-regret-weight",
                "1.0",
                "--p95-regret-target",
                "0.10",
                "--refresh-every",
                "6",
                "--select-best",
                "--validation-every",
                "4",
                "--validation-episodes",
                str(args.validation_episodes),
                "--validation-objective-episodes",
                str(max(64, args.batch_size)),
                "--selection-objective-weight",
                str(args.selection_objective_weight),
                "--selection-efficiency-target",
                str(args.selection_efficiency_target),
                "--selection-efficiency-weight",
                str(args.selection_efficiency_weight),
                "--grid",
                "2",
                "--history-draws",
                str(args.history_draws),
                "--exit-thresholds",
                args.exit_thresholds,
                "--chunk-size",
                "12",
                "--seed",
                str(args.seed + 700_000),
            ]
        )

    if not args.skip_audit:
        run_command(
            [
                sys.executable,
                "dynamic_continuation_audit.py",
                "--runs",
                str(stage1_dir),
                str(stage2_dir),
                "--episodes",
                str(args.audit_episodes),
                "--grid",
                "2",
                "--family",
                "history",
                "--history-draws",
                str(args.history_draws),
                "--exit-thresholds",
                args.exit_thresholds,
                "--chunk-size",
                "12",
                "--paired-draws",
                "--out-dir",
                str(audit_dir),
            ]
        )

    summary: dict[str, object] = {
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "audit_dir": str(audit_dir),
        "args": vars(args),
    }
    if (stage1_dir / "metrics.json").exists():
        summary["stage1_metrics"] = read_metrics(stage1_dir / "metrics.json")
    if (stage2_dir / "metrics.json").exists():
        summary["stage2_metrics"] = read_metrics(stage2_dir / "metrics.json")
    audit_summary = audit_dir / "continuation_audit_by_run.csv"
    if audit_summary.exists():
        summary["audit_table"] = str(audit_summary)
    (out_root / "two_stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
