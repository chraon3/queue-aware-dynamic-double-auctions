from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from analyze_queue_abandonment import (
    load_config,
    load_neural,
    make_draws,
    simulate_with_decomposition,
)
from dynamic_double_auction import QueueAwareMcAfeeMechanism, set_seed
from run_dynamic_robustness import SCENARIO_LABELS, SCENARIOS


METRICS = [
    "objective",
    "volume",
    "queue_length",
    "unmatched",
    "abandonment",
    "forced_share",
    "at_risk_abandon_rate",
    "mean_abandoner_opportunity",
]


def discover_runs(patterns: list[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        runs.extend(sorted(Path().glob(pattern)))
    return [path for path in runs if (path / "config.json").exists() and (path / "model.pt").exists()]


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    policies = ["Neural dynamic mechanism", "Payment-audited queue-aware McAfee"]
    rows: list[Dict[str, float | str | int]] = []
    for scenario, group in raw.groupby("scenario", sort=False):
        pivot = group.pivot(index="seed", columns="policy")
        row: Dict[str, float | str | int] = {
            "scenario": scenario,
            "label": str(group["label"].iloc[0]),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            neural = pivot[metric][policies[0]]
            queue = pivot[metric][policies[1]]
            gap = queue - neural
            row[f"neural_{metric}"] = float(neural.mean())
            row[f"queue_{metric}"] = float(queue.mean())
            row[f"gap_{metric}"] = float(gap.mean())
            row[f"gap_{metric}_std"] = float(gap.std(ddof=1))
        row["queue_objective_wins"] = int((pivot["objective"][policies[1]] > pivot["objective"][policies[0]]).sum())
        row["queue_lower_abandonment_wins"] = int((pivot["abandonment"][policies[1]] < pivot["abandonment"][policies[0]]).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["scenario_order"] = summary["scenario"].map({name: idx for idx, name in enumerate(SCENARIOS)})
    return summary.sort_values("scenario_order").drop(columns=["scenario_order"]).reset_index(drop=True)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    def fmt(value: float) -> str:
        if pd.isna(value):
            return "--"
        return f"{value:.3f}"

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Scenario & Neural obj. & Queue obj. & Obj. gap & Aband. gap & Obj. wins \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['label']} & "
            f"{fmt(row['neural_objective'])} & "
            f"{fmt(row['queue_objective'])} & "
            f"{fmt(row['gap_objective'])} & "
            f"{fmt(row['gap_abandonment'])} & "
            f"{int(row['queue_objective_wins'])}/{int(row['n_seeds'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--episodes", type=int, default=700)
    parser.add_argument("--draw-seed-offset", type=int, default=820000)
    parser.add_argument("--out-dir", default="experiments/queue_aware_stress")
    parser.add_argument("--paper-table", default="paper/tables/queue_aware_stress.tex")
    args = parser.parse_args()

    run_dirs = discover_runs(args.runs)
    if not run_dirs:
        raise FileNotFoundError("No dynamic run directories found.")

    rows = []
    for run_idx, run_dir in enumerate(run_dirs):
        base_cfg = load_config(run_dir / "config.json")
        for scenario_idx, (scenario, updates) in enumerate(SCENARIOS.items()):
            cfg = replace(base_cfg, eval_episodes=args.episodes, device="cpu", **updates)
            set_seed(cfg.seed + args.draw_seed_offset + scenario_idx)
            device = torch.device(cfg.device)
            draws = make_draws(cfg, args.episodes, args.draw_seed_offset + 1009 * run_idx + 37 * scenario_idx + cfg.seed, device)
            neural = load_neural(run_dir, cfg)
            queue = QueueAwareMcAfeeMechanism(cfg).to(device).eval()
            with torch.no_grad():
                neural_row = simulate_with_decomposition(neural, cfg, draws, "Neural dynamic mechanism", cfg.seed)
                queue_row = simulate_with_decomposition(queue, cfg, draws, "Payment-audited queue-aware McAfee", cfg.seed)
            for row in (neural_row, queue_row):
                row["run"] = run_dir.name
                row["scenario"] = scenario
                row["label"] = SCENARIO_LABELS[scenario]
            rows.extend([neural_row, queue_row])
            print(f"stress seed={cfg.seed} scenario={scenario}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw["scenario_order"] = raw["scenario"].map({name: idx for idx, name in enumerate(SCENARIOS)})
    raw = raw.sort_values(["scenario_order", "seed", "policy"]).drop(columns=["scenario_order"]).reset_index(drop=True)
    summary = summarize(raw)
    raw.to_csv(out_dir / "queue_aware_stress_by_seed.csv", index=False)
    summary.to_csv(out_dir / "queue_aware_stress_summary.csv", index=False)

    table_path = Path(args.paper_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_latex(summary, table_path)
    summary.to_csv(table_path.with_suffix(".csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
