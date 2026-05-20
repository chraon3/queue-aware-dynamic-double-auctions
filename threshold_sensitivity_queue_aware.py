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
from dynamic_double_auction import DynamicConfig, QueueAwareMcAfeeMechanism, set_seed, simulate_dynamic


def discover_runs(patterns: list[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        runs.extend(sorted(Path().glob(pattern)))
    return [path for path in runs if (path / "config.json").exists() and (path / "model.pt").exists()]


def threshold_grid() -> list[dict[str, float | int | str]]:
    return [
        {"setting": "Lower age", "age": 0.30, "imbalance": 0.25, "terminal": 2},
        {"setting": "Baseline", "age": 0.40, "imbalance": 0.25, "terminal": 2},
        {"setting": "Higher age", "age": 0.50, "imbalance": 0.25, "terminal": 2},
        {"setting": "Lower imbalance", "age": 0.40, "imbalance": 0.20, "terminal": 2},
        {"setting": "Higher imbalance", "age": 0.40, "imbalance": 0.30, "terminal": 2},
        {"setting": "Short terminal", "age": 0.40, "imbalance": 0.25, "terminal": 1},
        {"setting": "Long terminal", "age": 0.40, "imbalance": 0.25, "terminal": 3},
    ]


def cfg_with_thresholds(base: DynamicConfig, setting: dict[str, float | int | str], episodes: int) -> DynamicConfig:
    return replace(
        base,
        eval_episodes=episodes,
        device="cpu",
        queue_age_trigger=float(setting["age"]),
        queue_imbalance_trigger=float(setting["imbalance"]),
        queue_terminal_window=int(setting["terminal"]),
    )


def regret_metrics(cfg: DynamicConfig, episodes: int, seed: int) -> dict[str, float]:
    audit_cfg = replace(cfg, eval_episodes=episodes, seed=seed, device="cpu")
    set_seed(audit_cfg.seed)
    model = QueueAwareMcAfeeMechanism(audit_cfg).to(torch.device(audit_cfg.device)).eval()
    with torch.no_grad():
        sim = simulate_dynamic(model, audit_cfg, audit_cfg.eval_episodes, train=True)
    return {
        "mean_regret": float(sim["mean_regret"].item()),
        "p95_regret": float(sim["p95_regret"].item()),
        "max_regret": float(sim["max_regret"].item()),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, float | str | int]] = []
    order = {str(row["setting"]): idx for idx, row in enumerate(threshold_grid())}
    for setting, group in raw.groupby("setting", sort=False):
        row: Dict[str, float | str | int] = {
            "setting": setting,
            "age": float(group["age"].iloc[0]),
            "imbalance": float(group["imbalance"].iloc[0]),
            "terminal": int(group["terminal"].iloc[0]),
            "runs": int(group["seed"].nunique()),
            "setting_order": order[str(setting)],
        }
        for metric in [
            "queue_objective",
            "neural_objective",
            "objective_gap",
            "abandonment_gap",
            "volume_gap",
            "unmatched_gap",
            "mean_regret",
            "p95_regret",
            "max_regret",
        ]:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["objective_wins"] = int((group["objective_gap"] > 0.0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("setting_order").drop(columns=["setting_order"]).reset_index(drop=True)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    def fmt(value: float) -> str:
        if pd.isna(value):
            return "--"
        return f"{value:.3f}"

    rows = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Setting & $\\bar q_C$ & $\\iota_C$ & $h_C$ & Obj. gap & Aband. gap & Mean reg. & P95 reg. \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        rows.append(
            f"{row['setting']} & "
            f"{fmt(row['age'])} & "
            f"{fmt(row['imbalance'])} & "
            f"{int(row['terminal'])} & "
            f"{fmt(row['objective_gap'])} & "
            f"{fmt(row['abandonment_gap'])} & "
            f"{fmt(row['mean_regret'])} & "
            f"{fmt(row['p95_regret'])} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--regret-episodes", type=int, default=240)
    parser.add_argument("--draw-seed-offset", type=int, default=930000)
    parser.add_argument("--regret-seed-offset", type=int, default=940000)
    parser.add_argument("--out-dir", default="experiments/queue_trigger_sensitivity")
    parser.add_argument("--paper-table", default="paper/tables/queue_trigger_sensitivity.tex")
    args = parser.parse_args()

    run_dirs = discover_runs(args.runs)
    if not run_dirs:
        raise FileNotFoundError("No dynamic run directories found.")

    rows: list[dict[str, float | int | str]] = []
    settings = threshold_grid()
    for run_idx, run_dir in enumerate(run_dirs):
        base_cfg = load_config(run_dir / "config.json")
        neural_cfg = replace(base_cfg, eval_episodes=args.episodes, device="cpu")
        device = torch.device(neural_cfg.device)
        draws = make_draws(neural_cfg, args.episodes, args.draw_seed_offset + neural_cfg.seed, device)
        neural = load_neural(run_dir, neural_cfg)
        with torch.no_grad():
            neural_row = simulate_with_decomposition(
                neural,
                neural_cfg,
                draws,
                "Neural dynamic mechanism",
                neural_cfg.seed,
            )
        for setting_idx, setting in enumerate(settings):
            cfg = cfg_with_thresholds(base_cfg, setting, args.episodes)
            queue = QueueAwareMcAfeeMechanism(cfg).to(device).eval()
            with torch.no_grad():
                queue_row = simulate_with_decomposition(
                    queue,
                    cfg,
                    draws,
                    "Audited queue-aware McAfee",
                    cfg.seed,
                )
            regrets = regret_metrics(
                cfg,
                args.regret_episodes,
                args.regret_seed_offset + cfg.seed + 1009 * setting_idx + 7919 * run_idx,
            )
            row = {
                "run": run_dir.name,
                "seed": cfg.seed,
                "setting": str(setting["setting"]),
                "age": float(setting["age"]),
                "imbalance": float(setting["imbalance"]),
                "terminal": int(setting["terminal"]),
                "neural_objective": float(neural_row["objective"]),
                "queue_objective": float(queue_row["objective"]),
                "objective_gap": float(queue_row["objective"]) - float(neural_row["objective"]),
                "neural_abandonment": float(neural_row["abandonment"]),
                "queue_abandonment": float(queue_row["abandonment"]),
                "abandonment_gap": float(queue_row["abandonment"]) - float(neural_row["abandonment"]),
                "volume_gap": float(queue_row["volume"]) - float(neural_row["volume"]),
                "unmatched_gap": float(queue_row["unmatched"]) - float(neural_row["unmatched"]),
                **regrets,
            }
            rows.append(row)
            print(
                f"sensitivity seed={cfg.seed} setting={setting['setting']} "
                f"obj_gap={row['objective_gap']:.3f} aband_gap={row['abandonment_gap']:.3f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(out_dir / "queue_trigger_sensitivity_by_seed.csv", index=False)
    summary.to_csv(out_dir / "queue_trigger_sensitivity_summary.csv", index=False)
    write_latex(summary, Path(args.paper_table))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
