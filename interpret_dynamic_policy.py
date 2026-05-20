from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dynamic_double_auction import (
    DynamicConfig,
    add_arrivals,
    build_dynamic_model,
    compact_queue,
    dynamic_trade_count,
    efficient_queue_targets,
    make_reports,
    mechanism_forward,
    public_queue_features,
    queue_aware_mcafee_trade_count,
    queue_congestion_pressure,
)


SCENARIOS = [
    {
        "scenario": "balanced_high",
        "description": "Balanced high-surplus queue",
        "buyers": [0.90, 0.72, 0.55],
        "sellers": [0.12, 0.30, 0.48],
        "buyer_ages": [1, 1, 1],
        "seller_ages": [1, 1, 1],
        "t": 2,
    },
    {
        "scenario": "buyer_congested",
        "description": "Buyer-congested queue",
        "buyers": [0.95, 0.80, 0.65, 0.50],
        "sellers": [0.22, 0.42],
        "buyer_ages": [3, 2, 2, 1],
        "seller_ages": [1, 1],
        "t": 2,
    },
    {
        "scenario": "seller_congested",
        "description": "Seller-congested queue",
        "buyers": [0.88, 0.66],
        "sellers": [0.12, 0.28, 0.45, 0.60],
        "buyer_ages": [1, 1],
        "seller_ages": [3, 2, 2, 1],
        "t": 2,
    },
    {
        "scenario": "thin_market",
        "description": "Thin queue",
        "buyers": [0.80],
        "sellers": [0.35],
        "buyer_ages": [1],
        "seller_ages": [1],
        "t": 1,
    },
    {
        "scenario": "low_surplus",
        "description": "Low-surplus balanced queue",
        "buyers": [0.62, 0.52, 0.40],
        "sellers": [0.38, 0.50, 0.62],
        "buyer_ages": [1, 1, 1],
        "seller_ages": [1, 1, 1],
        "t": 2,
    },
    {
        "scenario": "late_balanced",
        "description": "Late balanced queue",
        "buyers": [0.90, 0.72, 0.55],
        "sellers": [0.12, 0.30, 0.48],
        "buyer_ages": [4, 3, 2],
        "seller_ages": [4, 3, 2],
        "t": 5,
    },
]


def load_config(path: Path) -> DynamicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = DynamicConfig.__dataclass_fields__.keys()
    return DynamicConfig(**{key: value for key, value in data.items() if key in valid})


def parse_run_dirs(text: str) -> list[Path]:
    paths = [Path(item.strip()) for item in text.split(",") if item.strip()]
    if not paths:
        raise ValueError("At least one run directory is required.")
    return paths


def fill_queue(values: list[float], ages: list[int], width: int, fill_value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_values = values[:width] + [fill_value] * max(width - len(values), 0)
    padded_ages = ages[:width] + [0] * max(width - len(ages), 0)
    active = [1.0] * min(len(values), width) + [0.0] * max(width - len(values), 0)
    return (
        torch.tensor([padded_values], dtype=torch.float32),
        torch.tensor([active], dtype=torch.float32),
        torch.tensor([padded_ages], dtype=torch.float32),
    )


def efficient_surplus(buyers: list[float], sellers: list[float]) -> float:
    b_sorted = sorted(buyers, reverse=True)
    s_sorted = sorted(sellers)
    return float(sum(max(b - s, 0.0) for b, s in zip(b_sorted, s_sorted)))


def scenario_metrics(model: torch.nn.Module, cfg: DynamicConfig, scenario: dict[str, Any], run_name: str) -> dict[str, Any]:
    buyer_values, buyer_active, buyer_ages = fill_queue(scenario["buyers"], scenario["buyer_ages"], cfg.max_buyers, 0.0)
    seller_costs, seller_active, seller_ages = fill_queue(scenario["sellers"], scenario["seller_ages"], cfg.max_sellers, 1.0)
    buyer_reports = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
    seller_reports = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
    t = min(int(scenario["t"]), cfg.horizon - 1)
    public_state = public_queue_features(buyer_active, seller_active, buyer_ages, seller_ages, t, cfg)
    with torch.no_grad():
        outcome = mechanism_forward(model, buyer_reports, seller_reports, public_state)

    match = outcome["match"]
    spread = buyer_reports[:, :, None] - seller_reports[:, None, :]
    active_pair = (buyer_active[:, :, None] * seller_active[:, None, :]).float()
    feasible_pair = ((spread >= 0.0).float() * active_pair)
    weighted_spread = match * torch.clamp(spread, min=0.0) * active_pair
    volume = float((match * active_pair).sum().item())
    surplus = float(weighted_spread.sum().item())
    buyer_payment = float((outcome["buyer_payments"] * buyer_active).sum().item())
    seller_transfer = float((outcome["seller_transfers"] * seller_active).sum().item())
    buyer_value_alloc = float(((match * active_pair) * buyer_reports[:, :, None]).sum().item())
    seller_cost_alloc = float(((match * active_pair) * seller_reports[:, None, :]).sum().item())
    buyer_utility = buyer_value_alloc - buyer_payment
    seller_utility = seller_transfer - seller_cost_alloc
    budget_wedge = buyer_payment - seller_transfer
    fb_surplus = efficient_surplus(scenario["buyers"], scenario["sellers"])
    fb_volume = dynamic_trade_count(scenario["buyers"], scenario["sellers"], "first_best")
    mcafee_volume = dynamic_trade_count(scenario["buyers"], scenario["sellers"], "mcafee")
    posted_volume = dynamic_trade_count(scenario["buyers"], scenario["sellers"], "posted")
    trade_reduction_volume = dynamic_trade_count(scenario["buyers"], scenario["sellers"], "trade_reduction")
    queue_mcafee_volume = queue_aware_mcafee_trade_count(
        list(zip(scenario["buyers"], scenario["buyer_ages"])),
        list(zip(scenario["sellers"], scenario["seller_ages"])),
        t,
        cfg,
    )
    denom = max(surplus, 1.0e-9)
    return {
        "run": run_name,
        "scenario": scenario["scenario"],
        "description": scenario["description"],
        "buyer_count": len(scenario["buyers"]),
        "seller_count": len(scenario["sellers"]),
        "mean_buyer_age": float(np.mean(scenario["buyer_ages"])) if scenario["buyer_ages"] else 0.0,
        "mean_seller_age": float(np.mean(scenario["seller_ages"])) if scenario["seller_ages"] else 0.0,
        "time": t,
        "first_best_volume": fb_volume,
        "mcafee_volume": mcafee_volume,
        "posted_volume": posted_volume,
        "trade_reduction_volume": trade_reduction_volume,
        "queue_mcafee_volume": queue_mcafee_volume,
        "neural_volume": volume,
        "first_best_surplus": fb_surplus,
        "neural_surplus": surplus,
        "neural_surplus_share": surplus / max(fb_surplus, 1.0e-9),
        "buyer_utility_share": buyer_utility / denom,
        "seller_utility_share": seller_utility / denom,
        "budget_wedge_share": budget_wedge / denom,
        "mean_buyer_payment_per_trade": buyer_payment / max(volume, 1.0e-9),
        "mean_seller_transfer_per_trade": seller_transfer / max(volume, 1.0e-9),
        "feasible_pair_count": float(feasible_pair.sum().item()),
    }


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "scenario",
        "first_best_volume",
        "mcafee_volume",
        "neural_volume",
        "neural_surplus_share",
        "buyer_utility_share",
        "seller_utility_share",
        "budget_wedge_share",
    ]
    labels = {
        "scenario": "Scenario",
        "first_best_volume": "FB vol.",
        "mcafee_volume": "McAfee",
        "neural_volume": "Neural",
        "neural_surplus_share": "Surplus",
        "buyer_utility_share": "Buyer",
        "seller_utility_share": "Seller",
        "budget_wedge_share": "Wedge",
    }
    rows = ["\\begin{tabular}{lrrrrrrr}", "\\toprule"]
    rows.append(" & ".join(labels[column] for column in columns) + " \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        values = [
            str(row["scenario"]).replace("_", " "),
            f"{row['first_best_volume']:.2f}",
            f"{row['mcafee_volume']:.2f}",
            f"{row['neural_volume']:.2f}",
            f"{row['neural_surplus_share']:.2f}",
            f"{row['buyer_utility_share']:.2f}",
            f"{row['seller_utility_share']:.2f}",
            f"{row['budget_wedge_share']:.2f}",
        ]
        rows.append(" & ".join(values) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    labels = [str(item).replace("_", "\n") for item in summary["scenario"]]
    x = np.arange(len(summary))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].bar(x - width, summary["first_best_volume"], width, label="First best", color="#3b6ea8")
    axes[0].bar(x, summary["mcafee_volume"], width, label="McAfee", color="#9c6b30")
    axes[0].bar(x + width, summary["neural_volume"], width, label="Neural", color="#2f8f6b")
    axes[0].set_ylabel("Expected trade volume")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0)
    axes[0].legend(frameon=False)
    axes[0].set_title("Trade intensity by queue state")

    axes[1].plot(x, summary["buyer_utility_share"], marker="o", label="Buyer utility", color="#3b6ea8")
    axes[1].plot(x, summary["seller_utility_share"], marker="o", label="Seller utility", color="#2f8f6b")
    axes[1].plot(x, summary["budget_wedge_share"], marker="o", label="Budget wedge", color="#8f3f3f")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Share of neural realized surplus")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=0)
    axes[1].legend(frameon=False)
    axes[1].set_title("Surplus split inside the mechanism")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def active_lists(values: torch.Tensor, active: torch.Tensor) -> list[float]:
    return [float(value) for value, is_active in zip(values.tolist(), active.tolist()) if is_active > 0.5]


def simulate_trace(model: torch.nn.Module, cfg: DynamicConfig, run_name: str, batch_size: int, seed_offset: int) -> pd.DataFrame:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed + seed_offset)
    buyer_values = torch.zeros(batch_size, cfg.max_buyers, device=device)
    seller_costs = torch.ones(batch_size, cfg.max_sellers, device=device)
    buyer_active = torch.zeros_like(buyer_values)
    seller_active = torch.zeros_like(seller_costs)
    buyer_ages = torch.zeros_like(buyer_values)
    seller_ages = torch.zeros_like(seller_costs)
    rows: list[pd.DataFrame] = []

    for t in range(cfg.horizon):
        buyer_values, buyer_active, buyer_ages = compact_queue(buyer_values, buyer_active, buyer_ages, 0.0)
        seller_costs, seller_active, seller_ages = compact_queue(seller_costs, seller_active, seller_ages, 1.0)
        buyer_values, buyer_active, buyer_ages = add_arrivals(buyer_values, buyer_active, buyer_ages, cfg.arrival_prob_buyer, 0.0, True)
        seller_costs, seller_active, seller_ages = add_arrivals(seller_costs, seller_active, seller_ages, cfg.arrival_prob_seller, 1.0, False)
        public_state = public_queue_features(buyer_active, seller_active, buyer_ages, seller_ages, t, cfg)
        buyer_reports, seller_reports = make_reports(buyer_values, seller_costs, buyer_active, seller_active)
        with torch.no_grad():
            outcome = mechanism_forward(model, buyer_reports, seller_reports, public_state)
        active_pair = buyer_active[:, :, None] * seller_active[:, None, :]
        match = outcome["match"] * active_pair
        pair_surplus = buyer_values[:, :, None] - seller_costs[:, None, :]
        neural_surplus = (match * pair_surplus).sum(dim=(1, 2))
        neural_volume = match.sum(dim=(1, 2))
        efficient_volume, efficient_surplus = efficient_queue_targets(buyer_values, seller_costs, buyer_active, seller_active)
        buyer_payment = (outcome["buyer_payments"] * buyer_active).sum(dim=1)
        seller_transfer = (outcome["seller_transfers"] * seller_active).sum(dim=1)
        buyer_value_alloc = (match * buyer_values[:, :, None]).sum(dim=(1, 2))
        seller_cost_alloc = (match * seller_costs[:, None, :]).sum(dim=(1, 2))
        buyer_utility = buyer_value_alloc - buyer_payment
        seller_utility = seller_transfer - seller_cost_alloc
        budget_wedge = buyer_payment - seller_transfer

        mcafee = []
        posted = []
        for idx in range(batch_size):
            buyers = active_lists(buyer_values[idx].detach().cpu(), buyer_active[idx].detach().cpu())
            sellers = active_lists(seller_costs[idx].detach().cpu(), seller_active[idx].detach().cpu())
            mcafee.append(dynamic_trade_count(buyers, sellers, "mcafee"))
            posted.append(dynamic_trade_count(buyers, sellers, "posted"))

        denom = torch.clamp(neural_surplus, min=1.0e-9)
        trace = pd.DataFrame(
            {
                "run": run_name,
                "period": t,
                "buyer_count": buyer_active.sum(dim=1).detach().cpu().numpy(),
                "seller_count": seller_active.sum(dim=1).detach().cpu().numpy(),
                "imbalance": public_state[:, 2].detach().cpu().numpy(),
                "queue_length": (buyer_active.sum(dim=1) + seller_active.sum(dim=1)).detach().cpu().numpy(),
                "mean_buyer_age": public_state[:, 3].detach().cpu().numpy() * max(cfg.max_patience, 1),
                "mean_seller_age": public_state[:, 4].detach().cpu().numpy() * max(cfg.max_patience, 1),
                "congestion_pressure": queue_congestion_pressure(public_state).detach().cpu().numpy(),
                "first_best_volume": efficient_volume.detach().cpu().numpy(),
                "mcafee_volume": np.array(mcafee, dtype=float),
                "posted_volume": np.array(posted, dtype=float),
                "neural_volume": neural_volume.detach().cpu().numpy(),
                "first_best_surplus": efficient_surplus.detach().cpu().numpy(),
                "neural_surplus": neural_surplus.detach().cpu().numpy(),
                "neural_surplus_share": (neural_surplus / torch.clamp(efficient_surplus, min=1.0e-9)).detach().cpu().numpy(),
                "buyer_payment": buyer_payment.detach().cpu().numpy(),
                "seller_transfer": seller_transfer.detach().cpu().numpy(),
                "buyer_utility": buyer_utility.detach().cpu().numpy(),
                "seller_utility": seller_utility.detach().cpu().numpy(),
                "budget_wedge": budget_wedge.detach().cpu().numpy(),
                "buyer_utility_share": (buyer_utility / denom).detach().cpu().numpy(),
                "seller_utility_share": (seller_utility / denom).detach().cpu().numpy(),
                "budget_wedge_share": (budget_wedge / denom).detach().cpu().numpy(),
            }
        )
        rows.append(trace)

        buyer_alloc = torch.clamp(match.sum(dim=2), 0.0, 1.0)
        seller_alloc = torch.clamp(match.sum(dim=1), 0.0, 1.0)
        unmatched_buyers = buyer_active * (1.0 - buyer_alloc)
        unmatched_sellers = seller_active * (1.0 - seller_alloc)
        depart_buyers = buyer_alloc.detach() > 0.5
        depart_sellers = seller_alloc.detach() > 0.5
        buyer_ages = buyer_ages + unmatched_buyers.detach()
        seller_ages = seller_ages + unmatched_sellers.detach()
        buyer_abandon_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * buyer_ages / max(cfg.max_patience, 1), 0.0, 1.0)
        seller_abandon_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * seller_ages / max(cfg.max_patience, 1), 0.0, 1.0)
        buyer_abandon = (torch.rand_like(buyer_active) < buyer_abandon_prob) | (buyer_ages >= cfg.max_patience)
        seller_abandon = (torch.rand_like(seller_active) < seller_abandon_prob) | (seller_ages >= cfg.max_patience)
        buyer_abandon = buyer_abandon & (buyer_active > 0.5) & (~depart_buyers)
        seller_abandon = seller_abandon & (seller_active > 0.5) & (~depart_sellers)
        buyer_active = torch.where(depart_buyers | buyer_abandon, torch.zeros_like(buyer_active), buyer_active)
        seller_active = torch.where(depart_sellers | seller_abandon, torch.zeros_like(seller_active), seller_active)
        buyer_values = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
        seller_costs = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
        buyer_ages = torch.where(buyer_active > 0.5, buyer_ages, torch.zeros_like(buyer_ages))
        seller_ages = torch.where(seller_active > 0.5, seller_ages, torch.zeros_like(seller_ages))

    return pd.concat(rows, ignore_index=True)


def trace_group_label(imbalance: float) -> str:
    if imbalance <= -0.125:
        return "seller heavy"
    if imbalance >= 0.125:
        return "buyer heavy"
    return "balanced"


def age_group_label(mean_queue_age: float) -> str:
    if mean_queue_age <= 0.25:
        return "fresh"
    if mean_queue_age <= 0.75:
        return "short wait"
    if mean_queue_age <= 1.25:
        return "medium wait"
    return "long wait"


def summarize_grouped_trace(trace: pd.DataFrame, group_col: str, order: dict[str, int]) -> pd.DataFrame:
    trace_numeric = [
        "first_best_volume",
        "mcafee_volume",
        "posted_volume",
        "neural_volume",
        "first_best_surplus",
        "neural_surplus",
        "buyer_payment",
        "seller_transfer",
        "buyer_utility",
        "seller_utility",
        "budget_wedge",
        "buyer_utility_share",
        "seller_utility_share",
        "queue_length",
        "mean_queue_age",
        "congestion_pressure",
    ]
    summary = trace.groupby(group_col, as_index=False)[trace_numeric].mean()
    summary["neural_surplus_share"] = summary["neural_surplus"] / summary["first_best_surplus"].clip(lower=1.0e-9)
    summary["buyer_utility_share"] = summary["buyer_utility"] / summary["neural_surplus"].clip(lower=1.0e-9)
    summary["seller_utility_share"] = summary["seller_utility"] / summary["neural_surplus"].clip(lower=1.0e-9)
    summary["budget_wedge_share"] = summary["budget_wedge"] / summary["neural_surplus"].clip(lower=1.0e-9)
    summary["buyer_payment_per_trade"] = summary["buyer_payment"] / summary["neural_volume"].clip(lower=1.0e-9)
    summary["seller_transfer_per_trade"] = summary["seller_transfer"] / summary["neural_volume"].clip(lower=1.0e-9)
    summary["budget_wedge_per_trade"] = summary["budget_wedge"] / summary["neural_volume"].clip(lower=1.0e-9)
    observations = trace.groupby(group_col).size().rename("observations").reset_index()
    summary = summary.merge(observations, on=group_col)
    summary = summary.rename(columns={group_col: "state"})
    return summary.sort_values("state", key=lambda col: col.map(order)).reset_index(drop=True)


def write_trace_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lrrrrrr}", "\\toprule"]
    rows.append("State & Obs. & FB vol. & McAfee & Neural & Surplus & Wedge \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        values = [
            str(row["state"]),
            f"{row['observations']:.0f}",
            f"{row['first_best_volume']:.2f}",
            f"{row['mcafee_volume']:.2f}",
            f"{row['neural_volume']:.2f}",
            f"{row['neural_surplus_share']:.2f}",
            f"{row['budget_wedge_share']:.2f}",
        ]
        rows.append(" & ".join(values) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_age_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule"]
    rows.append("Waiting & Obs. & Age & FB vol. & McAfee & Neural & Surplus & Buyer & Seller & Wedge \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        values = [
            str(row["state"]),
            f"{row['observations']:.0f}",
            f"{row['mean_queue_age']:.2f}",
            f"{row['first_best_volume']:.2f}",
            f"{row['mcafee_volume']:.2f}",
            f"{row['neural_volume']:.2f}",
            f"{row['neural_surplus_share']:.2f}",
            f"{row['buyer_utility_share']:.2f}",
            f"{row['seller_utility_share']:.2f}",
            f"{row['budget_wedge_share']:.2f}",
        ]
        rows.append(" & ".join(values) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_trace_summary(summary: pd.DataFrame, path: Path) -> None:
    order = ["seller heavy", "balanced", "buyer heavy"]
    frame = summary.set_index("state").reindex(order).reset_index()
    x = np.arange(len(frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width, frame["first_best_volume"], width, label="First best", color="#3b6ea8")
    ax.bar(x, frame["mcafee_volume"], width, label="McAfee", color="#9c6b30")
    ax.bar(x + width, frame["neural_volume"], width, label="Neural", color="#2f8f6b")
    ax.set_xticks(x)
    ax.set_xticklabels([label.title() for label in frame["state"]])
    ax.set_ylabel("Mean trade volume on simulated paths")
    ax.set_title("Mechanism behavior on visited queue states")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_age_summary(summary: pd.DataFrame, path: Path) -> None:
    order = ["fresh", "short wait", "medium wait", "long wait"]
    frame = summary.set_index("state").reindex(order).reset_index()
    x = np.arange(len(frame))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].bar(x - width, frame["first_best_volume"], width, label="First best", color="#3b6ea8")
    axes[0].bar(x, frame["mcafee_volume"], width, label="McAfee", color="#9c6b30")
    axes[0].bar(x + width, frame["neural_volume"], width, label="Neural", color="#2f8f6b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(label).title() for label in frame["state"]])
    axes[0].set_ylabel("Mean trade volume")
    axes[0].set_title("Trade volume by waiting pressure")
    axes[0].legend(frameon=False)

    axes[1].plot(x, frame["buyer_utility_share"], marker="o", label="Buyer utility", color="#3b6ea8")
    axes[1].plot(x, frame["seller_utility_share"], marker="o", label="Seller utility", color="#2f8f6b")
    axes[1].plot(x, frame["budget_wedge_share"], marker="o", label="Budget wedge", color="#8f3f3f")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(label).title() for label in frame["state"]])
    axes[1].set_ylabel("Share of neural realized surplus")
    axes[1].set_title("Surplus split by waiting pressure")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=",".join(f"experiments/dynamic_patience_value_seed{seed}" for seed in [81, 83, 89, 91, 93, 97, 101, 103, 107, 109]))
    parser.add_argument("--out-dir", default="experiments/dynamic_policy_interpretation")
    parser.add_argument("--paper-table", default="paper/tables/dynamic_policy_interpretation.tex")
    parser.add_argument("--paper-figure", default="paper/figures/dynamic_policy_interpretation.png")
    parser.add_argument("--trace-batch-size", type=int, default=512)
    parser.add_argument("--trace-seed-offset", type=int, default=360000)
    parser.add_argument("--paper-trace-table", default="paper/tables/dynamic_policy_trace.tex")
    parser.add_argument("--paper-trace-figure", default="paper/figures/dynamic_policy_trace.png")
    parser.add_argument("--paper-age-table", default="paper/tables/dynamic_policy_age_trace.tex")
    parser.add_argument("--paper-age-figure", default="paper/figures/dynamic_policy_age_trace.png")
    args = parser.parse_args()

    run_dirs = parse_run_dirs(args.runs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        cfg = load_config(run_dir / "config.json")
        device = torch.device(cfg.device)
        model = build_dynamic_model(cfg).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        model.eval()
        for scenario in SCENARIOS:
            rows.append(scenario_metrics(model, cfg, scenario, run_dir.name))

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "dynamic_policy_interpretation_by_run.csv", index=False)
    group_cols = ["scenario", "description"]
    numeric_cols = [column for column in detail.columns if column not in {"run", "scenario", "description"}]
    summary = detail.groupby(group_cols, as_index=False)[numeric_cols].mean()
    scenario_order = {scenario["scenario"]: idx for idx, scenario in enumerate(SCENARIOS)}
    summary = summary.sort_values("scenario", key=lambda col: col.map(scenario_order)).reset_index(drop=True)
    summary.to_csv(out_dir / "dynamic_policy_interpretation_summary.csv", index=False)
    Path(args.paper_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.paper_figure).parent.mkdir(parents=True, exist_ok=True)
    Path(args.paper_trace_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.paper_trace_figure).parent.mkdir(parents=True, exist_ok=True)
    Path(args.paper_age_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.paper_age_figure).parent.mkdir(parents=True, exist_ok=True)
    write_latex_table(summary, Path(args.paper_table))
    plot_summary(summary, Path(args.paper_figure))

    trace_tables: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        cfg = load_config(run_dir / "config.json")
        device = torch.device(cfg.device)
        model = build_dynamic_model(cfg).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        model.eval()
        trace_tables.append(simulate_trace(model, cfg, run_dir.name, args.trace_batch_size, args.trace_seed_offset))
    trace = pd.concat(trace_tables, ignore_index=True)
    trace["state"] = trace["imbalance"].map(trace_group_label)
    trace["mean_queue_age"] = 0.5 * (trace["mean_buyer_age"] + trace["mean_seller_age"])
    trace["age_state"] = trace["mean_queue_age"].map(age_group_label)
    trace.to_csv(out_dir / "dynamic_policy_trace_by_period.csv", index=False)
    state_order = {"seller heavy": 0, "balanced": 1, "buyer heavy": 2}
    trace_summary = summarize_grouped_trace(trace, "state", state_order)
    trace_summary.to_csv(out_dir / "dynamic_policy_trace_summary.csv", index=False)
    write_trace_table(trace_summary, Path(args.paper_trace_table))
    plot_trace_summary(trace_summary, Path(args.paper_trace_figure))
    age_order = {"fresh": 0, "short wait": 1, "medium wait": 2, "long wait": 3}
    age_summary = summarize_grouped_trace(trace, "age_state", age_order)
    age_summary.to_csv(out_dir / "dynamic_policy_age_trace_summary.csv", index=False)
    write_age_table(age_summary, Path(args.paper_age_table))
    plot_age_summary(age_summary, Path(args.paper_age_figure))
    print(summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
