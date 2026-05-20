from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from dynamic_double_auction import (
    DynamicConfig,
    add_arrivals,
    compact_queue,
    evaluate_dynamic_baselines,
    make_reports,
    set_seed,
)
from econpinn_double_auction import grid_regret, utilities


@dataclass
class SoftBaselineConfig:
    max_buyers: int = 4
    max_sellers: int = 4
    horizon: int = 6
    batch_size: int = 96
    train_steps: int = 160
    lr: float = 2.0e-3
    hidden: int = 64
    depth: int = 3
    feature_mode: str = "ranked"
    arrival_prob_buyer: float = 0.75
    arrival_prob_seller: float = 0.75
    wait_cost: float = 0.02
    discount: float = 0.97
    max_patience: int = 5
    abandon_base: float = 0.01
    abandon_slope: float = 0.08
    regret_grid: int = 5
    regret_weight: float = 2.0
    budget_weight: float = 12.0
    ir_weight: float = 12.0
    no_deficit_target: float = 0.0
    eval_episodes: int = 700
    seed: int = 201
    log_every: int = 40
    out_dir: str = "experiments/soft_dynamic_seed201"
    device: str = "cpu"


class SoftPaymentDoubleAuction(nn.Module):
    """
    RegretNet-style payment baseline.

    Matching is softly normalized for comparability, but transfers are not
    parameterized to satisfy truthful IR or budget balance by construction.
    Those economic restrictions enter only through penalty terms.
    """

    def __init__(self, hidden: int = 64, depth: int = 3, feature_mode: str = "ranked") -> None:
        super().__init__()
        self.feature_mode = feature_mode
        in_dim = 10 if feature_mode == "ranked" else 6
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, buyer_reports: torch.Tensor, seller_reports: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = buyer_reports[:, :, None]
        a = seller_reports[:, None, :]
        spread = b - a
        mean_b = buyer_reports.mean(dim=1, keepdim=True)[:, :, None].expand_as(spread)
        mean_a = seller_reports.mean(dim=1, keepdim=True)[:, None, :].expand_as(spread)
        tightness = mean_b - mean_a
        features = [b.expand_as(spread), a.expand_as(spread), spread, mean_b, mean_a, tightness]
        if self.feature_mode == "ranked":
            batch, n_buyers = buyer_reports.shape
            n_sellers = seller_reports.shape[1]
            buyer_order = torch.argsort(buyer_reports, dim=1, descending=True)
            seller_order = torch.argsort(seller_reports, dim=1, descending=False)
            buyer_rank = torch.empty_like(buyer_order, dtype=buyer_reports.dtype)
            seller_rank = torch.empty_like(seller_order, dtype=seller_reports.dtype)
            buyer_rank.scatter_(1, buyer_order, torch.arange(n_buyers, device=buyer_reports.device, dtype=buyer_reports.dtype).expand(batch, n_buyers))
            seller_rank.scatter_(1, seller_order, torch.arange(n_sellers, device=seller_reports.device, dtype=seller_reports.dtype).expand(batch, n_sellers))
            buyer_rank = buyer_rank / max(n_buyers - 1, 1)
            seller_rank = seller_rank / max(n_sellers - 1, 1)
            buyer_rank_pair = buyer_rank[:, :, None].expand_as(spread)
            seller_rank_pair = seller_rank[:, None, :].expand_as(spread)
            features.extend([buyer_rank_pair, seller_rank_pair, seller_rank_pair - buyer_rank_pair, spread - tightness])

        out = self.net(torch.stack(features, dim=-1))
        weights = torch.sigmoid(out[..., 0]) * (spread >= 0.0).float()
        match = weights
        for _ in range(4):
            match = match / torch.clamp(match.sum(dim=2, keepdim=True), min=1.0)
            match = match / torch.clamp(match.sum(dim=1, keepdim=True), min=1.0)

        buyer_prices = torch.sigmoid(out[..., 1])
        seller_transfers = torch.sigmoid(out[..., 2])
        buyer_payments = (match * buyer_prices).sum(dim=2)
        seller_payments = (match * seller_transfers).sum(dim=1)
        buyer_alloc = match.sum(dim=2)
        seller_alloc = match.sum(dim=1)
        return {
            "match": match,
            "buyer_alloc": buyer_alloc,
            "seller_alloc": seller_alloc,
            "buyer_payments": buyer_payments,
            "seller_transfers": seller_payments,
        }


def to_dynamic_config(cfg: SoftBaselineConfig) -> DynamicConfig:
    valid = DynamicConfig.__dataclass_fields__.keys()
    data = {key: value for key, value in asdict(cfg).items() if key in valid}
    return DynamicConfig(**data)


def constraint_violations(
    outcome: Dict[str, torch.Tensor],
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    buyer_u, seller_u = utilities(outcome, buyer_values, seller_costs)
    row_violation = torch.relu(outcome["match"].sum(dim=2) - 1.0).mean()
    col_violation = torch.relu(outcome["match"].sum(dim=1) - 1.0).mean()
    budget_violation = torch.relu(outcome["seller_transfers"].sum(dim=1) - outcome["buyer_payments"].sum(dim=1)).mean()
    buyer_ir_violation = torch.relu(-buyer_u).mean()
    seller_ir_violation = torch.relu(-seller_u).mean()
    return {
        "capacity_violation": row_violation + col_violation,
        "budget_violation": budget_violation,
        "buyer_ir_violation": buyer_ir_violation,
        "seller_ir_violation": seller_ir_violation,
        "total_ir_violation": buyer_ir_violation + seller_ir_violation,
    }


def simulate_soft_dynamic(
    model: SoftPaymentDoubleAuction,
    cfg: SoftBaselineConfig,
    batch_size: int,
    train: bool,
) -> Dict[str, torch.Tensor]:
    device = torch.device(cfg.device)
    dyn_cfg = to_dynamic_config(cfg)
    buyer_values = torch.zeros(batch_size, cfg.max_buyers, device=device)
    seller_costs = torch.ones(batch_size, cfg.max_sellers, device=device)
    buyer_active = torch.zeros_like(buyer_values)
    seller_active = torch.zeros_like(seller_costs)
    buyer_ages = torch.zeros_like(buyer_values)
    seller_ages = torch.zeros_like(seller_costs)

    rewards = []
    surpluses = []
    waits = []
    volumes = []
    abandons = []
    buyer_regrets = []
    seller_regrets = []
    active_regret_samples = []
    violation_rows = []

    for t in range(cfg.horizon):
        buyer_values, buyer_active, buyer_ages = compact_queue(buyer_values, buyer_active, buyer_ages, 0.0)
        seller_costs, seller_active, seller_ages = compact_queue(seller_costs, seller_active, seller_ages, 1.0)
        buyer_values, buyer_active, buyer_ages = add_arrivals(buyer_values, buyer_active, buyer_ages, cfg.arrival_prob_buyer, 0.0, True)
        seller_costs, seller_active, seller_ages = add_arrivals(seller_costs, seller_active, seller_ages, cfg.arrival_prob_seller, 1.0, False)

        buyer_reports, seller_reports = make_reports(buyer_values, seller_costs, buyer_active, seller_active)
        out = model(buyer_reports, seller_reports)
        active_pair = buyer_active[:, :, None] * seller_active[:, None, :]
        match = out["match"] * active_pair
        masked_out = dict(out)
        masked_out["match"] = match
        masked_out["buyer_alloc"] = torch.clamp(match.sum(dim=2), 0.0, 1.0)
        masked_out["seller_alloc"] = torch.clamp(match.sum(dim=1), 0.0, 1.0)

        pair_surplus = buyer_values[:, :, None] - seller_costs[:, None, :]
        surplus = (match * pair_surplus).sum(dim=(1, 2))
        unmatched_buyers = buyer_active * (1.0 - masked_out["buyer_alloc"])
        unmatched_sellers = seller_active * (1.0 - masked_out["seller_alloc"])
        wait_penalty = cfg.wait_cost * (unmatched_buyers.sum(dim=1) + unmatched_sellers.sum(dim=1))
        reward = surplus - wait_penalty
        rewards.append((cfg.discount**t) * reward)
        surpluses.append(surplus)
        waits.append(wait_penalty)
        volumes.append(match.sum(dim=(1, 2)))
        violation_rows.append(constraint_violations(masked_out, buyer_values, seller_costs))

        if train:
            b_reg, s_reg = grid_regret(model, buyer_reports, seller_reports, cfg.regret_grid)
            active_b_reg = b_reg * buyer_active
            active_s_reg = s_reg * seller_active
            buyer_regrets.append(active_b_reg.sum() / torch.clamp(buyer_active.sum(), min=1.0))
            seller_regrets.append(active_s_reg.sum() / torch.clamp(seller_active.sum(), min=1.0))
            if torch.any(buyer_active > 0.5):
                active_regret_samples.append(active_b_reg[buyer_active > 0.5])
            if torch.any(seller_active > 0.5):
                active_regret_samples.append(active_s_reg[seller_active > 0.5])

        depart_buyers = masked_out["buyer_alloc"].detach() > 0.5
        depart_sellers = masked_out["seller_alloc"].detach() > 0.5
        buyer_ages = buyer_ages + unmatched_buyers.detach()
        seller_ages = seller_ages + unmatched_sellers.detach()
        buyer_abandon_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * buyer_ages / max(cfg.max_patience, 1), 0.0, 1.0)
        seller_abandon_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * seller_ages / max(cfg.max_patience, 1), 0.0, 1.0)
        buyer_abandon = (torch.rand_like(buyer_active) < buyer_abandon_prob) | (buyer_ages >= cfg.max_patience)
        seller_abandon = (torch.rand_like(seller_active) < seller_abandon_prob) | (seller_ages >= cfg.max_patience)
        buyer_abandon = buyer_abandon & (buyer_active > 0.5) & (~depart_buyers)
        seller_abandon = seller_abandon & (seller_active > 0.5) & (~depart_sellers)
        abandons.append(buyer_abandon.float().sum(dim=1) + seller_abandon.float().sum(dim=1))

        buyer_active = torch.where(depart_buyers | buyer_abandon, torch.zeros_like(buyer_active), buyer_active)
        seller_active = torch.where(depart_sellers | seller_abandon, torch.zeros_like(seller_active), seller_active)
        buyer_values = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
        seller_costs = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
        buyer_ages = torch.where(buyer_active > 0.5, buyer_ages, torch.zeros_like(buyer_ages))
        seller_ages = torch.where(seller_active > 0.5, seller_ages, torch.zeros_like(seller_ages))

    mean_regret = torch.tensor(0.0, device=device)
    p95_regret = torch.tensor(0.0, device=device)
    max_regret = torch.tensor(0.0, device=device)
    if buyer_regrets:
        mean_regret = torch.stack(buyer_regrets).mean() + torch.stack(seller_regrets).mean()
    if active_regret_samples:
        all_regrets = torch.cat([values.reshape(-1) for values in active_regret_samples])
        p95_regret = torch.quantile(all_regrets, 0.95)
        max_regret = all_regrets.max()

    violation_summary = {
        key: torch.stack([row[key] for row in violation_rows]).mean()
        for key in violation_rows[0]
    }
    total_reward = torch.stack(rewards).sum(dim=0)
    return {
        "objective": total_reward.mean(),
        "mean_surplus": torch.stack(surpluses).mean(),
        "mean_wait_cost": torch.stack(waits).mean(),
        "mean_volume": torch.stack(volumes).mean(),
        "mean_abandonment": torch.stack(abandons).mean(),
        "mean_regret": mean_regret,
        "p95_regret": p95_regret,
        "max_regret": max_regret,
        **violation_summary,
    }


def train_soft_baseline(cfg: SoftBaselineConfig) -> Tuple[SoftPaymentDoubleAuction, pd.DataFrame, Dict[str, float]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SoftPaymentDoubleAuction(cfg.hidden, cfg.depth, cfg.feature_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.0e-4)
    history = []

    for step in range(1, cfg.train_steps + 1):
        sim = simulate_soft_dynamic(model, cfg, cfg.batch_size, train=True)
        loss = (
            -sim["objective"]
            + cfg.regret_weight * sim["mean_regret"]
            + cfg.budget_weight * sim["budget_violation"]
            + cfg.ir_weight * sim["total_ir_violation"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            row = {key: float(value.item()) for key, value in sim.items()}
            row["step"] = step
            row["loss"] = float(loss.item())
            history.append(row)
            print(
                f"step={step:04d} obj={row['objective']:.4f} regret={row['mean_regret']:.4f} "
                f"budget_v={row['budget_violation']:.4f} ir_v={row['total_ir_violation']:.4f}",
                flush=True,
            )

    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    model.eval()
    with torch.no_grad():
        eval_sim = simulate_soft_dynamic(model, replace(cfg, regret_grid=max(cfg.regret_grid, 7)), cfg.eval_episodes, train=True)
    metrics = {key: float(value.item()) for key, value in eval_sim.items()}
    dyn_cfg = to_dynamic_config(cfg)
    metrics.update(evaluate_dynamic_baselines(dyn_cfg))
    first_best = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    metrics["dynamic_soft_efficiency"] = metrics["objective"] / first_best
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(hist, out_dir)
    return model, hist, metrics


def plot_history(history: pd.DataFrame, out_dir: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes = axes.ravel()
    axes[0].plot(history["step"], history["objective"], color="#0f766e")
    axes[0].set_title("Soft baseline objective")
    axes[1].plot(history["step"], history["mean_regret"], color="#b91c1c")
    axes[1].set_title("Grid regret")
    axes[2].plot(history["step"], history["budget_violation"], color="#ea580c", label="budget")
    axes[2].plot(history["step"], history["total_ir_violation"], color="#6d28d9", label="IR")
    axes[2].set_title("Soft constraint violations")
    axes[2].legend(frameon=False)
    axes[3].plot(history["step"], history["p95_regret"], color="#334155")
    axes[3].set_title("Tail regret")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "soft_training_curves.png")
    plt.close(fig)


def suite() -> list[SoftBaselineConfig]:
    base = SoftBaselineConfig()
    return [
        replace(base, seed=201, out_dir="experiments/soft_dynamic_seed201"),
        replace(base, seed=203, out_dir="experiments/soft_dynamic_seed203"),
        replace(base, seed=207, out_dir="experiments/soft_dynamic_seed207"),
        replace(base, seed=211, out_dir="experiments/soft_dynamic_seed211"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=SoftBaselineConfig.seed)
    parser.add_argument("--out-dir", default=SoftBaselineConfig.out_dir)
    args = parser.parse_args()

    configs = suite() if args.suite else [replace(SoftBaselineConfig(), seed=args.seed, out_dir=args.out_dir)]
    manifest = []
    for cfg in configs:
        if args.skip_existing and (Path(cfg.out_dir) / "metrics.json").exists():
            print(f"Skipping existing {cfg.out_dir}", flush=True)
            continue
        print(f"\n=== Running {cfg.out_dir} ===", flush=True)
        _, _, metrics = train_soft_baseline(cfg)
        manifest.append({"config": asdict(cfg), "metrics": metrics})
    Path("experiments/soft_dynamic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
