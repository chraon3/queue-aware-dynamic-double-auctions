from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from dynamic_continuation_audit import (
    build_strategy_set,
    continuation_regret_for_side,
    make_base_draws,
    parse_exit_thresholds,
)
from dynamic_double_auction import DynamicConfig, QueueAwareMcAfeeMechanism, set_seed


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def audit_seed(
    seed: int,
    episodes: int,
    grid: int,
    radius: float,
    family: str,
    history_draws: int,
    exit_thresholds: torch.Tensor,
    chunk_size: int,
    seed_offset: int,
    cfg_template: DynamicConfig,
) -> dict[str, float | str]:
    cfg = DynamicConfig(**{**cfg_template.__dict__, "seed": seed, "eval_episodes": episodes, "mechanism": "queue_mcafee"})
    set_seed(cfg.seed + seed_offset)
    device = torch.device(cfg.device)
    model = QueueAwareMcAfeeMechanism(cfg).to(device)
    coefficients, strategy_exit_thresholds, report_strategy_count = build_strategy_set(
        family,
        radius,
        grid,
        history_draws,
        exit_thresholds,
        cfg.seed + seed_offset,
        device,
    )
    with torch.no_grad():
        buyer_draws = make_base_draws(cfg, episodes, cfg.seed + seed_offset + 11, device)
        seller_draws = make_base_draws(cfg, episodes, cfg.seed + seed_offset + 29, device)
        buyer_regret, buyer_no_exit_regret = continuation_regret_for_side(
            model,
            cfg,
            buyer_draws,
            "buyer",
            coefficients,
            strategy_exit_thresholds,
            chunk_size,
        )
        seller_regret, seller_no_exit_regret = continuation_regret_for_side(
            model,
            cfg,
            seller_draws,
            "seller",
            coefficients,
            strategy_exit_thresholds,
            chunk_size,
        )
    all_regret = torch.cat([buyer_regret, seller_regret])
    no_exit_regret = torch.cat([buyer_no_exit_regret, seller_no_exit_regret])
    return {
        "run": f"queue_aware_mcafee_seed{seed}",
        "policy": "Payment-audited queue-aware McAfee",
        "audit_family": family,
        "seed": float(seed),
        "seed_offset": float(seed_offset),
        "episodes_per_side": float(episodes),
        "grid": float(grid),
        "radius": float(radius),
        "report_strategy_count": float(report_strategy_count),
        "exit_threshold_count": float(exit_thresholds.numel()),
        "continuation_strategy_count": float(coefficients.shape[0]),
        "history_draws": float(history_draws if family.startswith("history") else 0),
        "buyer_continuation_mean_regret": float(buyer_regret.mean().item()),
        "seller_continuation_mean_regret": float(seller_regret.mean().item()),
        "continuation_mean_regret": float(all_regret.mean().item()),
        "continuation_p95_regret": float(torch.quantile(all_regret, 0.95).item()),
        "continuation_max_regret": float(all_regret.max().item()),
        "no_exit_continuation_mean_regret": float(no_exit_regret.mean().item()),
        "no_exit_continuation_p95_regret": float(torch.quantile(no_exit_regret, 0.95).item()),
        "no_exit_continuation_max_regret": float(no_exit_regret.max().item()),
        "continuation_regret_count": float(all_regret.numel()),
    }


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lrrrr}", "\\toprule"]
    rows.append("Policy & Mean & P95 & Max & No-exit mean \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["policy"]),
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
    parser.add_argument("--seeds", default="101,103,107,109,81,83,89,91,93,97")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--family", choices=["age", "state", "history", "history_nonlinear", "history_piecewise"], default="history")
    parser.add_argument("--history-draws", type=int, default=24)
    parser.add_argument("--exit-thresholds", default="2,4,99")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--seed-offset", type=int, default=80_000)
    parser.add_argument("--seed-stride", type=int, default=997)
    parser.add_argument("--out-dir", default="experiments/queue_aware_mcafee_continuation_audit")
    parser.add_argument("--paper-table", default="paper/tables/queue_aware_continuation_audit.tex")
    args = parser.parse_args()

    cfg_template = DynamicConfig(
        max_buyers=4,
        max_sellers=4,
        horizon=6,
        eval_episodes=args.episodes,
        regret_grid=7,
        wait_cost=0.02,
        arrival_prob_buyer=0.75,
        arrival_prob_seller=0.75,
        max_patience=5,
        abandon_base=0.01,
        abandon_slope=0.08,
        mechanism="queue_mcafee",
    )
    seeds = parse_seeds(args.seeds)
    exit_thresholds = parse_exit_thresholds(args.exit_thresholds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, seed in enumerate(seeds):
        print(f"Continuation auditing queue-aware McAfee seed={seed}", flush=True)
        rows.append(
            audit_seed(
                seed,
                args.episodes,
                args.grid,
                args.radius,
                args.family,
                args.history_draws,
                exit_thresholds,
                args.chunk_size,
                args.seed_offset + idx * args.seed_stride,
                cfg_template,
            )
        )
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(out_dir / "queue_aware_continuation_by_seed.csv", index=False)
    numeric_columns = [
        "continuation_mean_regret",
        "continuation_p95_regret",
        "continuation_max_regret",
        "no_exit_continuation_mean_regret",
        "no_exit_continuation_p95_regret",
        "no_exit_continuation_max_regret",
    ]
    summary = by_seed.groupby("policy", as_index=False)[numeric_columns].mean()
    summary.to_csv(out_dir / "queue_aware_continuation_summary.csv", index=False)
    paper_table = Path(args.paper_table)
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    write_latex_table(summary, paper_table)
    print(by_seed.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
