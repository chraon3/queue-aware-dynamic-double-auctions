from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from analyze_queue_abandonment import add_arrivals_from_draws, load_config, make_draws
from dynamic_double_auction import (
    DynamicConfig,
    QueueAwareMcAfeeMechanism,
    build_dynamic_model,
    compact_queue,
    make_reports,
    mechanism_forward,
    public_queue_features,
    queue_pressure_trigger,
    set_seed,
)


def load_neural(run_dir: Path, cfg: DynamicConfig) -> nn.Module:
    device = torch.device(cfg.device)
    model = build_dynamic_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return model


def mcafee_trade_count(values: list[float], costs: list[float]) -> int:
    b_sorted = sorted(values, reverse=True)
    s_sorted = sorted(costs)
    kmax = min(len(b_sorted), len(s_sorted))
    efficient_k = 0
    for idx in range(kmax):
        if b_sorted[idx] >= s_sorted[idx]:
            efficient_k = idx + 1
    if efficient_k <= 0:
        return 0
    trade_k = max(efficient_k - 1, 0)
    if efficient_k < len(b_sorted) and efficient_k < len(s_sorted):
        price = 0.5 * (b_sorted[efficient_k] + s_sorted[efficient_k])
        if s_sorted[efficient_k - 1] <= price <= b_sorted[efficient_k - 1]:
            trade_k = efficient_k
    return trade_k


def uniform_price(values: list[float], costs: list[float], trade_k: int) -> float:
    if trade_k <= 0:
        return math.nan
    b_sorted = sorted(values, reverse=True)
    s_sorted = sorted(costs)
    last_b = b_sorted[trade_k - 1]
    last_s = s_sorted[trade_k - 1]
    if trade_k < len(b_sorted) and trade_k < len(s_sorted):
        price = 0.5 * (b_sorted[trade_k] + s_sorted[trade_k])
        return float(min(max(price, last_s), last_b))
    return float(0.5 * (last_b + last_s))


def queue_trade_count(values: list[float], costs: list[float], public_state_row: torch.Tensor, cfg: DynamicConfig) -> tuple[int, int, int, bool]:
    b_sorted = sorted(values, reverse=True)
    s_sorted = sorted(costs)
    kmax = min(len(b_sorted), len(s_sorted))
    efficient_k = 0
    for idx in range(kmax):
        if b_sorted[idx] >= s_sorted[idx]:
            efficient_k = idx + 1
    mcafee_k = mcafee_trade_count(values, costs)
    if efficient_k <= 0:
        return 0, mcafee_k, efficient_k, False
    mean_age_pressure = 0.5 * (public_state_row[3] + public_state_row[4])
    imbalance = torch.abs(public_state_row[2])
    triggered = bool(queue_pressure_trigger(mean_age_pressure[None], imbalance[None], public_state_row[8][None], cfg).item())
    return (efficient_k if triggered else mcafee_k), mcafee_k, efficient_k, triggered


def init_sums(policy: str, seed: int) -> dict[str, float | str | int]:
    return {
        "seed": seed,
        "policy": policy,
        "states": 0.0,
        "traded_states": 0.0,
        "volume": 0.0,
        "surplus": 0.0,
        "buyer_payment": 0.0,
        "seller_transfer": 0.0,
        "buyer_ir": 0.0,
        "seller_ir": 0.0,
        "wedge": 0.0,
        "opportunity_states": 0.0,
        "triggered_opportunities": 0.0,
        "relaxed_states": 0.0,
        "extra_trades": 0.0,
        "extra_surplus": 0.0,
        "price_shift_sum": 0.0,
        "price_shift_count": 0.0,
    }


def simulate_price_impact(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: dict[str, torch.Tensor],
    policy: str,
    seed: int,
    collect_relaxation: bool,
) -> dict[str, float | str | int]:
    device = torch.device(cfg.device)
    episodes = draws["buyer_arrival"].shape[1]
    buyer_values = torch.zeros(episodes, cfg.max_buyers, device=device)
    seller_costs = torch.ones(episodes, cfg.max_sellers, device=device)
    buyer_active = torch.zeros_like(buyer_values)
    seller_active = torch.zeros_like(seller_costs)
    buyer_ages = torch.zeros_like(buyer_values)
    seller_ages = torch.zeros_like(seller_costs)
    sums = init_sums(policy, seed)

    for t in range(cfg.horizon):
        buyer_values, buyer_active, buyer_ages = compact_queue(buyer_values, buyer_active, buyer_ages, 0.0)
        seller_costs, seller_active, seller_ages = compact_queue(seller_costs, seller_active, seller_ages, 1.0)
        buyer_values, buyer_active, buyer_ages = add_arrivals_from_draws(
            buyer_values,
            buyer_active,
            buyer_ages,
            draws["buyer_arrival"][t],
            draws["buyer_value"][t],
            cfg.arrival_prob_buyer,
            0.0,
        )
        seller_costs, seller_active, seller_ages = add_arrivals_from_draws(
            seller_costs,
            seller_active,
            seller_ages,
            draws["seller_arrival"][t],
            draws["seller_cost"][t],
            cfg.arrival_prob_seller,
            1.0,
        )

        public_state = public_queue_features(buyer_active, seller_active, buyer_ages, seller_ages, t, cfg)
        buyer_reports, seller_reports = make_reports(buyer_values, seller_costs, buyer_active, seller_active)
        out = mechanism_forward(model, buyer_reports, seller_reports, public_state)
        active_pair = buyer_active[:, :, None] * seller_active[:, None, :]
        match = out["match"] * active_pair
        buyer_alloc = torch.clamp(match.sum(dim=2), 0.0, 1.0)
        seller_alloc = torch.clamp(match.sum(dim=1), 0.0, 1.0)
        volume = match.sum(dim=(1, 2))
        pair_surplus = buyer_values[:, :, None] - seller_costs[:, None, :]
        surplus = (match * pair_surplus).sum(dim=(1, 2))
        buyer_payment = (out["buyer_payments"] * buyer_active).sum(dim=1)
        seller_transfer = (out["seller_transfers"] * seller_active).sum(dim=1)
        buyer_value_alloc = (buyer_values * buyer_alloc).sum(dim=1)
        seller_cost_alloc = (seller_costs * seller_alloc).sum(dim=1)
        buyer_ir = buyer_value_alloc - buyer_payment
        seller_ir = seller_transfer - seller_cost_alloc
        wedge = buyer_payment - seller_transfer

        sums["states"] = float(sums["states"]) + float(episodes)
        sums["traded_states"] = float(sums["traded_states"]) + float((volume > 1.0e-9).float().sum().item())
        sums["volume"] = float(sums["volume"]) + float(volume.sum().item())
        sums["surplus"] = float(sums["surplus"]) + float(surplus.sum().item())
        sums["buyer_payment"] = float(sums["buyer_payment"]) + float(buyer_payment.sum().item())
        sums["seller_transfer"] = float(sums["seller_transfer"]) + float(seller_transfer.sum().item())
        sums["buyer_ir"] = float(sums["buyer_ir"]) + float(buyer_ir.sum().item())
        sums["seller_ir"] = float(sums["seller_ir"]) + float(seller_ir.sum().item())
        sums["wedge"] = float(sums["wedge"]) + float(wedge.sum().item())

        if collect_relaxation:
            buyer_np = buyer_values.detach().cpu()
            seller_np = seller_costs.detach().cpu()
            b_active_np = buyer_active.detach().cpu()
            s_active_np = seller_active.detach().cpu()
            public_cpu = public_state.detach().cpu()
            for row in range(episodes):
                values = [float(x) for x, flag in zip(buyer_np[row].tolist(), b_active_np[row].tolist()) if flag > 0.5]
                costs = [float(x) for x, flag in zip(seller_np[row].tolist(), s_active_np[row].tolist()) if flag > 0.5]
                qa_k, mcafee_k, efficient_k, triggered = queue_trade_count(values, costs, public_cpu[row], cfg)
                if efficient_k > mcafee_k:
                    sums["opportunity_states"] = float(sums["opportunity_states"]) + 1.0
                    if triggered:
                        sums["triggered_opportunities"] = float(sums["triggered_opportunities"]) + 1.0
                if qa_k > mcafee_k:
                    sums["relaxed_states"] = float(sums["relaxed_states"]) + 1.0
                    extra = qa_k - mcafee_k
                    sums["extra_trades"] = float(sums["extra_trades"]) + float(extra)
                    b_sorted = sorted(values, reverse=True)
                    s_sorted = sorted(costs)
                    for rank in range(mcafee_k, qa_k):
                        sums["extra_surplus"] = float(sums["extra_surplus"]) + max(b_sorted[rank] - s_sorted[rank], 0.0)
                    qa_price = uniform_price(values, costs, qa_k)
                    mc_price = uniform_price(values, costs, mcafee_k)
                    if not math.isnan(qa_price) and not math.isnan(mc_price):
                        sums["price_shift_sum"] = float(sums["price_shift_sum"]) + qa_price - mc_price
                        sums["price_shift_count"] = float(sums["price_shift_count"]) + 1.0

        depart_buyers = buyer_alloc.detach() > 0.5
        depart_sellers = seller_alloc.detach() > 0.5
        unmatched_buyers = buyer_active * (1.0 - buyer_alloc.detach())
        unmatched_sellers = seller_active * (1.0 - seller_alloc.detach())
        buyer_ages_after_wait = buyer_ages + unmatched_buyers
        seller_ages_after_wait = seller_ages + unmatched_sellers
        buyer_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * buyer_ages_after_wait / max(cfg.max_patience, 1), 0.0, 1.0)
        seller_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * seller_ages_after_wait / max(cfg.max_patience, 1), 0.0, 1.0)
        buyer_forced = buyer_ages_after_wait >= cfg.max_patience
        seller_forced = seller_ages_after_wait >= cfg.max_patience
        buyer_stochastic = (draws["buyer_abandon"][t] < buyer_prob) & (~buyer_forced)
        seller_stochastic = (draws["seller_abandon"][t] < seller_prob) & (~seller_forced)
        buyer_abandon = (buyer_forced | buyer_stochastic) & (buyer_active > 0.5) & (~depart_buyers)
        seller_abandon = (seller_forced | seller_stochastic) & (seller_active > 0.5) & (~depart_sellers)

        buyer_active = torch.where(depart_buyers | buyer_abandon, torch.zeros_like(buyer_active), buyer_active)
        seller_active = torch.where(depart_sellers | seller_abandon, torch.zeros_like(seller_active), seller_active)
        buyer_values = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
        seller_costs = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
        buyer_ages = torch.where(buyer_active > 0.5, buyer_ages_after_wait, torch.zeros_like(buyer_ages))
        seller_ages = torch.where(seller_active > 0.5, seller_ages_after_wait, torch.zeros_like(seller_ages))

    return sums


def ratios(row: dict[str, Any]) -> dict[str, Any]:
    volume = max(float(row["volume"]), 1.0e-9)
    surplus = max(float(row["surplus"]), 1.0e-9)
    has_opportunity = float(row["opportunity_states"]) > 0.0
    has_extra = float(row["extra_trades"]) > 0.0
    has_price_shift = float(row["price_shift_count"]) > 0.0
    opportunity = max(float(row["opportunity_states"]), 1.0e-9)
    extra = max(float(row["extra_trades"]), 1.0e-9)
    price_shift_count = max(float(row["price_shift_count"]), 1.0e-9)
    return {
        **row,
        "volume_per_state": float(row["volume"]) / max(float(row["states"]), 1.0e-9),
        "traded_state_share": float(row["traded_states"]) / max(float(row["states"]), 1.0e-9),
        "buyer_payment_per_trade": float(row["buyer_payment"]) / volume,
        "seller_transfer_per_trade": float(row["seller_transfer"]) / volume,
        "buyer_ir_per_trade": float(row["buyer_ir"]) / volume,
        "seller_ir_per_trade": float(row["seller_ir"]) / volume,
        "budget_wedge_per_surplus": float(row["wedge"]) / surplus,
        "relaxed_opportunity_share": float(row["relaxed_states"]) / opportunity if has_opportunity else float("nan"),
        "triggered_opportunity_share": float(row["triggered_opportunities"]) / opportunity if has_opportunity else float("nan"),
        "extra_surplus_per_trade": float(row["extra_surplus"]) / extra if has_extra else float("nan"),
        "price_shift_vs_mcafee": float(row["price_shift_sum"]) / price_shift_count if has_price_shift else float("nan"),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "volume_per_state",
        "traded_state_share",
        "buyer_payment_per_trade",
        "seller_transfer_per_trade",
        "buyer_ir_per_trade",
        "seller_ir_per_trade",
        "budget_wedge_per_surplus",
        "relaxed_opportunity_share",
        "triggered_opportunity_share",
        "extra_surplus_per_trade",
        "price_shift_vs_mcafee",
    ]
    rows = []
    for policy, group in frame.groupby("policy", sort=False):
        row: dict[str, Any] = {"policy": policy, "runs": len(group)}
        for col in metrics:
            values = group[col].replace([float("inf"), -float("inf")], pd.NA).dropna()
            row[col] = float(values.mean()) if len(values) else float("nan")
            row[col + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: float, blank_nan: bool = False) -> str:
    if pd.isna(value):
        return "--" if blank_nan else "nan"
    return f"{value:.3f}"


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule"]
    rows.append("Policy & Vol. & Pay/tr. & Tr./tr. & B-IR & S-IR & Wedge & Relax & Extra S & $\\Delta p$ \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["policy"]).replace("Payment-audited queue-aware McAfee", "Audited queue McAfee").replace("Neural dynamic mechanism", "Neural"),
                    fmt(row["volume_per_state"]),
                    fmt(row["buyer_payment_per_trade"]),
                    fmt(row["seller_transfer_per_trade"]),
                    fmt(row["buyer_ir_per_trade"]),
                    fmt(row["seller_ir_per_trade"]),
                    fmt(row["budget_wedge_per_surplus"]),
                    fmt(row["relaxed_opportunity_share"], True),
                    fmt(row["extra_surplus_per_trade"], True),
                    fmt(row["price_shift_vs_mcafee"], True),
                ]
            )
            + " \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_dirs_from_pattern(pattern: str) -> list[Path]:
    return sorted(Path(".").glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="experiments/dynamic_patience_value_seed*")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--draw-seed-offset", type=int, default=310000)
    parser.add_argument("--out-dir", default="experiments/queue_aware_price_impact")
    parser.add_argument("--paper-table", default="paper/tables/queue_aware_price_impact.tex")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in run_dirs_from_pattern(args.run_glob):
        config_path = run_dir / "config.json"
        model_path = run_dir / "model.pt"
        if not config_path.exists() or not model_path.exists():
            continue
        cfg = load_config(config_path)
        cfg = DynamicConfig(**{**cfg.__dict__, "eval_episodes": args.episodes})
        device = torch.device(cfg.device)
        set_seed(cfg.seed)
        draws = make_draws(cfg, args.episodes, cfg.seed + args.draw_seed_offset, device)
        neural = load_neural(run_dir, cfg)
        queue = QueueAwareMcAfeeMechanism(cfg).to(device).eval()
        with torch.no_grad():
            rows.append(ratios(simulate_price_impact(neural, cfg, draws, "Neural dynamic mechanism", cfg.seed, False)))
            rows.append(ratios(simulate_price_impact(queue, cfg, draws, "Payment-audited queue-aware McAfee", cfg.seed, True)))

    by_seed = pd.DataFrame(rows)
    summary = summarize(by_seed)
    by_seed.to_csv(out_dir / "price_impact_by_seed.csv", index=False)
    summary.to_csv(out_dir / "price_impact_summary.csv", index=False)
    paper_table = Path(args.paper_table)
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    write_latex(summary, paper_table)
    manifest = {
        "run_glob": args.run_glob,
        "episodes": args.episodes,
        "draw_seed_offset": args.draw_seed_offset,
        "runs": int(by_seed["seed"].nunique()) if not by_seed.empty else 0,
        "out_dir": str(out_dir),
        "paper_table": str(paper_table),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
