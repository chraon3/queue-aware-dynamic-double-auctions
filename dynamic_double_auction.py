from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from econpinn_double_auction import (
    Config as StaticConfig,
    HardConstrainedDoubleAuction,
    adversarial_regret,
    grid_regret,
    mcafee_welfare_np,
    posted_price_welfare_np,
    trade_reduction_welfare_np,
    utilities,
)


@dataclass
class DynamicConfig:
    max_buyers: int = 4
    max_sellers: int = 4
    horizon: int = 8
    batch_size: int = 192
    train_steps: int = 500
    lr: float = 2.0e-3
    hidden: int = 64
    depth: int = 3
    feature_mode: str = "ranked"
    mechanism: str = "base"
    arrival_prob_buyer: float = 0.72
    arrival_prob_seller: float = 0.72
    wait_cost: float = 0.015
    discount: float = 0.97
    max_patience: int = 5
    abandon_base: float = 0.015
    abandon_slope: float = 0.12
    regret_grid: int = 7
    regret_method: str = "grid"
    adv_steps: int = 4
    adv_lr: float = 0.7
    adv_restarts: int = 1
    regret_weight: float = 4.0
    regret_target: float = 0.01
    augmented_rho: float = 6.0
    congestion_aux_weight: float = 0.0
    congestion_volume_weight: float = 0.75
    imitation_aux_weight: float = 0.0
    value_hidden: int = 64
    value_loss_weight: float = 0.25
    value_refine_steps: int = 0
    value_refine_batch_size: int = 256
    value_refine_lr: float = 1.0e-3
    value_refine_td_weight: float = 0.5
    value_refine_log_every: int = 50
    queue_age_trigger: float = 0.40
    queue_imbalance_trigger: float = 0.25
    queue_terminal_window: int = 2
    eval_episodes: int = 1000
    eval_regret_samples: int = 256
    select_best: bool = False
    selection_regret_penalty: float = 3.0
    selection_min_step: int = 1
    seed: int = 101
    log_every: int = 50
    out_dir: str = "experiments/dynamic_queue"
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ContinuationValueNet(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class StateAwareHardConstrainedDoubleAuction(nn.Module):
    """Hard-constrained pairwise mechanism with public dynamic state features."""

    supports_state = True

    def __init__(self, hidden: int = 64, depth: int = 3, feature_mode: str = "ranked", state_dim: int = 12) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        self.state_dim = state_dim
        width_in = (6 if feature_mode == "basic" else 10) + state_dim
        layers = []
        for layer_idx in range(depth):
            layers.append(nn.Linear(width_in if layer_idx == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        buyer_reports: torch.Tensor,
        seller_reports: torch.Tensor,
        public_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch, n_buyers = buyer_reports.shape
        n_sellers = seller_reports.shape[1]
        device = buyer_reports.device
        if public_state is None:
            public_state = torch.zeros(batch, self.state_dim, device=device, dtype=buyer_reports.dtype)
        if public_state.shape[0] != batch:
            if batch % public_state.shape[0] != 0:
                raise ValueError("public_state batch dimension must match or divide report batch dimension.")
            public_state = public_state.repeat_interleave(batch // public_state.shape[0], dim=0)

        b = buyer_reports[:, :, None].expand(batch, n_buyers, n_sellers)
        a = seller_reports[:, None, :].expand(batch, n_buyers, n_sellers)
        spread = b - a
        mean_b = buyer_reports.mean(dim=1, keepdim=True)[:, :, None].expand_as(b)
        mean_a = seller_reports.mean(dim=1, keepdim=True)[:, None, :].expand_as(a)
        market_tightness = mean_b - mean_a
        feasible = (spread >= 0).float()
        features = self._features(buyer_reports, seller_reports, b, a, spread, mean_b, mean_a, market_tightness)
        state_pair = public_state[:, None, None, :].expand(batch, n_buyers, n_sellers, self.state_dim)
        features = torch.cat([torch.stack(features, dim=-1), state_pair], dim=-1)
        raw = self.net(features)
        alloc_logit = raw[..., 0]
        seller_share_raw = raw[..., 1]
        budget_wedge_raw = raw[..., 2]

        weights = torch.sigmoid(alloc_logit) * feasible
        row_sum = weights.sum(dim=2, keepdim=True)
        weights = weights / torch.clamp(row_sum, min=1.0)
        col_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.clamp(col_sum, min=1.0)

        nonneg_spread = torch.clamp(spread, min=0.0)
        seller_share = torch.sigmoid(seller_share_raw)
        wedge_share = torch.sigmoid(budget_wedge_raw)
        buyer_share = seller_share + (1.0 - seller_share) * wedge_share
        seller_unit_transfer = a + seller_share * nonneg_spread
        buyer_unit_payment = a + buyer_share * nonneg_spread

        buyer_alloc = weights.sum(dim=2)
        seller_alloc = weights.sum(dim=1)
        buyer_payments = (weights * buyer_unit_payment).sum(dim=2)
        seller_transfers = (weights * seller_unit_transfer).sum(dim=1)
        return {
            "match": weights,
            "buyer_alloc": buyer_alloc,
            "seller_alloc": seller_alloc,
            "buyer_payments": buyer_payments,
            "seller_transfers": seller_transfers,
            "buyer_unit_payment": buyer_unit_payment,
            "seller_unit_transfer": seller_unit_transfer,
        }

    def _features(
        self,
        buyer_reports: torch.Tensor,
        seller_reports: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        spread: torch.Tensor,
        mean_b: torch.Tensor,
        mean_a: torch.Tensor,
        market_tightness: torch.Tensor,
    ) -> list[torch.Tensor]:
        base = [b, a, spread, mean_b, mean_a, market_tightness]
        if self.feature_mode == "basic":
            return base
        if self.feature_mode != "ranked":
            raise ValueError(f"Unknown feature mode: {self.feature_mode}")
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
        buyer_rank_pair = buyer_rank[:, :, None].expand_as(b)
        seller_rank_pair = seller_rank[:, None, :].expand_as(a)
        rank_gap = seller_rank_pair - buyer_rank_pair
        centered_spread = spread - market_tightness
        return base + [buyer_rank_pair, seller_rank_pair, rank_gap, centered_spread]


class QueueAwareMcAfeeMechanism(nn.Module):
    """Auditable heuristic that uses queue congestion to relax McAfee reduction."""

    supports_state = True

    def __init__(self, cfg: DynamicConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        buyer_reports: torch.Tensor,
        seller_reports: torch.Tensor,
        public_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch, n_buyers = buyer_reports.shape
        n_sellers = seller_reports.shape[1]
        if public_state is None:
            public_state = torch.zeros(batch, 12, device=buyer_reports.device, dtype=buyer_reports.dtype)
        if public_state.shape[0] != batch:
            public_state = public_state.repeat_interleave(batch // public_state.shape[0], dim=0)
        match = torch.zeros(batch, n_buyers, n_sellers, device=buyer_reports.device, dtype=buyer_reports.dtype)
        buyer_payments = torch.zeros(batch, n_buyers, device=buyer_reports.device, dtype=buyer_reports.dtype)
        seller_transfers = torch.zeros(batch, n_sellers, device=buyer_reports.device, dtype=buyer_reports.dtype)
        for row in range(batch):
            buyers = buyer_reports[row]
            sellers = seller_reports[row]
            buyer_order = torch.argsort(buyers, descending=True)
            seller_order = torch.argsort(sellers, descending=False)
            sorted_buyers = buyers[buyer_order]
            sorted_sellers = sellers[seller_order]
            feasible = sorted_buyers - sorted_sellers
            efficient_k = int((feasible >= 0.0).sum().item())
            trade_k = 0
            if efficient_k > 0:
                mcafee_k = max(efficient_k - 1, 0)
                if efficient_k < n_buyers and efficient_k < n_sellers:
                    price = 0.5 * (sorted_buyers[efficient_k] + sorted_sellers[efficient_k])
                    if sorted_sellers[efficient_k - 1] <= price <= sorted_buyers[efficient_k - 1]:
                        mcafee_k = efficient_k
                mean_age_pressure = 0.5 * (public_state[row, 3] + public_state[row, 4])
                imbalance = torch.abs(public_state[row, 2])
                late_market = public_state[row, 8] >= terminal_pressure_cutoff(self.cfg)
                congested = bool(
                    (mean_age_pressure >= self.cfg.queue_age_trigger)
                    or (imbalance >= self.cfg.queue_imbalance_trigger)
                    or late_market
                )
                trade_k = efficient_k if congested else mcafee_k
            if trade_k <= 0:
                continue
            last_b = sorted_buyers[trade_k - 1]
            last_s = sorted_sellers[trade_k - 1]
            if trade_k < n_buyers and trade_k < n_sellers:
                price = 0.5 * (sorted_buyers[trade_k] + sorted_sellers[trade_k])
                price = torch.clamp(price, min=last_s, max=last_b)
            else:
                price = 0.5 * (last_b + last_s)
            for rank in range(trade_k):
                b_idx = buyer_order[rank]
                s_idx = seller_order[rank]
                match[row, b_idx, s_idx] = 1.0
                buyer_payments[row, b_idx] = price
                seller_transfers[row, s_idx] = price
        buyer_alloc = match.sum(dim=2)
        seller_alloc = match.sum(dim=1)
        buyer_unit_payment = torch.zeros_like(match)
        seller_unit_transfer = torch.zeros_like(match)
        return {
            "match": match,
            "buyer_alloc": buyer_alloc,
            "seller_alloc": seller_alloc,
            "buyer_payments": buyer_payments,
            "seller_transfers": seller_transfers,
            "buyer_unit_payment": buyer_unit_payment,
            "seller_unit_transfer": seller_unit_transfer,
        }


def build_dynamic_model(cfg: DynamicConfig) -> nn.Module:
    if cfg.mechanism == "base":
        return HardConstrainedDoubleAuction(cfg.hidden, cfg.depth, cfg.feature_mode)
    if cfg.mechanism == "state":
        return StateAwareHardConstrainedDoubleAuction(cfg.hidden, cfg.depth, cfg.feature_mode)
    raise ValueError(f"Unknown mechanism type: {cfg.mechanism}")


def mechanism_forward(
    model: nn.Module,
    buyer_reports: torch.Tensor,
    seller_reports: torch.Tensor,
    public_state: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if getattr(model, "supports_state", False):
        return model(buyer_reports, seller_reports, public_state)
    return model(buyer_reports, seller_reports)


def compact_queue(
    values: torch.Tensor,
    active: torch.Tensor,
    ages: torch.Tensor,
    fill_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move active agents to the front of each queue after departures."""
    batch, width = values.shape
    new_values = torch.full_like(values, fill_value)
    new_active = torch.zeros_like(active)
    new_ages = torch.zeros_like(ages)
    for idx in range(batch):
        keep_mask = active[idx] > 0.5
        keep = values[idx][keep_mask]
        keep_ages = ages[idx][keep_mask]
        count = min(int(keep.numel()), width)
        if count:
            new_values[idx, :count] = keep[:count]
            new_active[idx, :count] = 1.0
            new_ages[idx, :count] = keep_ages[:count]
    return new_values, new_active, new_ages


def add_arrivals(
    values: torch.Tensor,
    active: torch.Tensor,
    ages: torch.Tensor,
    arrival_prob: float,
    fill_value: float,
    is_buyer: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, width = values.shape
    for idx in range(batch):
        if torch.rand((), device=values.device).item() <= arrival_prob:
            open_slots = torch.nonzero(active[idx] < 0.5, as_tuple=False).flatten()
            if open_slots.numel() > 0:
                slot = int(open_slots[0].item())
                draw = torch.rand((), device=values.device)
                values[idx, slot] = draw if is_buyer else draw
                active[idx, slot] = 1.0
                ages[idx, slot] = 0.0
    values = torch.where(active > 0.5, values, torch.full_like(values, fill_value))
    ages = torch.where(active > 0.5, ages, torch.zeros_like(ages))
    return values, active, ages


def state_features(
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    buyer_active: torch.Tensor,
    seller_active: torch.Tensor,
    buyer_ages: torch.Tensor,
    seller_ages: torch.Tensor,
    t: int,
    cfg: DynamicConfig,
) -> torch.Tensor:
    buyer_count = buyer_active.sum(dim=1)
    seller_count = seller_active.sum(dim=1)
    active_b_values = buyer_values * buyer_active
    active_s_costs = seller_costs * seller_active
    mean_b = active_b_values.sum(dim=1) / torch.clamp(buyer_count, min=1.0)
    mean_s = active_s_costs.sum(dim=1) / torch.clamp(seller_count, min=1.0)
    max_b = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values)).max(dim=1).values
    min_s = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs)).min(dim=1).values
    mean_b_age = (buyer_ages * buyer_active).sum(dim=1) / torch.clamp(buyer_count, min=1.0)
    mean_s_age = (seller_ages * seller_active).sum(dim=1) / torch.clamp(seller_count, min=1.0)
    total_capacity = float(cfg.max_buyers + cfg.max_sellers)
    t_frac = torch.full_like(buyer_count, float(t) / max(cfg.horizon - 1, 1))
    return torch.stack(
        [
            buyer_count / max(cfg.max_buyers, 1),
            seller_count / max(cfg.max_sellers, 1),
            (buyer_count - seller_count) / total_capacity,
            mean_b,
            mean_s,
            max_b,
            min_s,
            max_b - min_s,
            mean_b_age / max(cfg.max_patience, 1),
            mean_s_age / max(cfg.max_patience, 1),
            t_frac,
            (buyer_count + seller_count) / total_capacity,
        ],
        dim=1,
    )


def public_queue_features(
    buyer_active: torch.Tensor,
    seller_active: torch.Tensor,
    buyer_ages: torch.Tensor,
    seller_ages: torch.Tensor,
    t: int,
    cfg: DynamicConfig,
) -> torch.Tensor:
    buyer_count = buyer_active.sum(dim=1)
    seller_count = seller_active.sum(dim=1)
    total_capacity = float(max(cfg.max_buyers + cfg.max_sellers, 1))
    mean_b_age = (buyer_ages * buyer_active).sum(dim=1) / torch.clamp(buyer_count, min=1.0)
    mean_s_age = (seller_ages * seller_active).sum(dim=1) / torch.clamp(seller_count, min=1.0)
    max_b_age = torch.where(buyer_active > 0.5, buyer_ages, torch.zeros_like(buyer_ages)).max(dim=1).values
    max_s_age = torch.where(seller_active > 0.5, seller_ages, torch.zeros_like(seller_ages)).max(dim=1).values
    t_frac = torch.full_like(buyer_count, float(t) / max(cfg.horizon - 1, 1))
    buyer_pressure = torch.clamp(cfg.abandon_base + cfg.abandon_slope * mean_b_age / max(cfg.max_patience, 1), 0.0, 1.0)
    seller_pressure = torch.clamp(cfg.abandon_base + cfg.abandon_slope * mean_s_age / max(cfg.max_patience, 1), 0.0, 1.0)
    return torch.stack(
        [
            buyer_count / max(cfg.max_buyers, 1),
            seller_count / max(cfg.max_sellers, 1),
            (buyer_count - seller_count) / total_capacity,
            mean_b_age / max(cfg.max_patience, 1),
            mean_s_age / max(cfg.max_patience, 1),
            max_b_age / max(cfg.max_patience, 1),
            max_s_age / max(cfg.max_patience, 1),
            (mean_b_age - mean_s_age) / max(cfg.max_patience, 1),
            t_frac,
            (buyer_count + seller_count) / total_capacity,
            buyer_pressure,
            seller_pressure,
        ],
        dim=1,
    )


def make_reports(
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    buyer_active: torch.Tensor,
    seller_active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    buyer_reports = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
    seller_reports = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
    return buyer_reports, seller_reports


def efficient_queue_targets(
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    buyer_active: torch.Tensor,
    seller_active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Myopic efficient trade count and surplus for the active queue."""
    k = min(buyer_values.shape[1], seller_costs.shape[1])
    inactive_buyers = torch.full_like(buyer_values, -1.0)
    inactive_sellers = torch.full_like(seller_costs, 2.0)
    buyers_sorted = torch.sort(torch.where(buyer_active > 0.5, buyer_values, inactive_buyers), dim=1, descending=True).values
    sellers_sorted = torch.sort(torch.where(seller_active > 0.5, seller_costs, inactive_sellers), dim=1, descending=False).values
    spreads = buyers_sorted[:, :k] - sellers_sorted[:, :k]
    efficient_spreads = torch.clamp(spreads, min=0.0)
    efficient_volume = (spreads >= 0.0).float().sum(dim=1)
    efficient_surplus = efficient_spreads.sum(dim=1)
    return efficient_volume, efficient_surplus


def queue_congestion_pressure(public_state: torch.Tensor) -> torch.Tensor:
    """Smooth public-state pressure used only as a training auxiliary weight."""
    age_pressure = 0.5 * (public_state[:, 3] + public_state[:, 4])
    imbalance = torch.abs(public_state[:, 2])
    total_queue = public_state[:, 9]
    time_pressure = public_state[:, 8]
    return torch.clamp(0.45 * total_queue + 0.25 * age_pressure + 0.20 * imbalance + 0.10 * time_pressure, 0.0, 1.0)


def terminal_pressure_cutoff(cfg: DynamicConfig) -> float:
    """Normalized time cutoff for the public terminal trigger."""
    return 1.0 - float(cfg.queue_terminal_window) / max(cfg.horizon - 1, 1)


def queue_pressure_trigger(
    mean_age_pressure: torch.Tensor,
    imbalance: torch.Tensor,
    t_frac: torch.Tensor,
    cfg: DynamicConfig,
) -> torch.Tensor:
    return (
        (mean_age_pressure >= cfg.queue_age_trigger)
        | (torch.abs(imbalance) >= cfg.queue_imbalance_trigger)
        | (t_frac >= terminal_pressure_cutoff(cfg))
    )


def queue_aware_imitation_match(
    buyer_reports: torch.Tensor,
    seller_reports: torch.Tensor,
    public_state: torch.Tensor,
    cfg: DynamicConfig,
) -> torch.Tensor:
    """Queue-aware McAfee allocation target for auxiliary distillation."""
    with torch.no_grad():
        target = QueueAwareMcAfeeMechanism(cfg).to(buyer_reports.device)
        return target(buyer_reports, seller_reports, public_state)["match"].detach()


def dynamic_regret(
    model: nn.Module,
    buyer_reports: torch.Tensor,
    seller_reports: torch.Tensor,
    cfg: DynamicConfig,
    public_state: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if getattr(model, "supports_state", False):
        return state_grid_regret(model, buyer_reports, seller_reports, public_state, cfg.regret_grid)
    if cfg.regret_method == "grid":
        return grid_regret(model, buyer_reports, seller_reports, cfg.regret_grid)
    if cfg.regret_method == "adv":
        with torch.enable_grad():
            return adversarial_regret(model, buyer_reports, seller_reports, cfg.adv_steps, cfg.adv_lr, cfg.adv_restarts)
    if cfg.regret_method == "hybrid":
        grid_b, grid_s = grid_regret(model, buyer_reports, seller_reports, cfg.regret_grid)
        with torch.enable_grad():
            adv_b, adv_s = adversarial_regret(model, buyer_reports, seller_reports, cfg.adv_steps, cfg.adv_lr, cfg.adv_restarts)
        return torch.maximum(grid_b, adv_b), torch.maximum(grid_s, adv_s)
    raise ValueError(f"Unknown dynamic regret method: {cfg.regret_method}")


def state_grid_regret(
    model: nn.Module,
    buyer_reports: torch.Tensor,
    seller_reports: torch.Tensor,
    public_state: torch.Tensor | None,
    grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = buyer_reports.device
    batch, n_buyers = buyer_reports.shape
    n_sellers = seller_reports.shape[1]
    reports_grid = torch.linspace(0.0, 1.0, grid_size, device=device)
    truthful_outcome = mechanism_forward(model, buyer_reports, seller_reports, public_state)
    truthful_buyer_u, truthful_seller_u = utilities(truthful_outcome, buyer_reports, seller_reports)
    buyer_regrets = []
    seller_regrets = []
    for i in range(n_buyers):
        b_rep = buyer_reports.repeat_interleave(grid_size, dim=0)
        s_rep = seller_reports.repeat_interleave(grid_size, dim=0)
        b_rep[:, i] = reports_grid.repeat(batch)
        out = mechanism_forward(model, b_rep, s_rep, public_state)
        u_i, _ = utilities(
            out,
            buyer_reports.repeat_interleave(grid_size, dim=0),
            seller_reports.repeat_interleave(grid_size, dim=0),
        )
        best_i = torch.maximum(truthful_buyer_u[:, i], u_i[:, i].reshape(batch, grid_size).max(dim=1).values)
        buyer_regrets.append(torch.relu(best_i - truthful_buyer_u[:, i]))
    for j in range(n_sellers):
        b_rep = buyer_reports.repeat_interleave(grid_size, dim=0)
        s_rep = seller_reports.repeat_interleave(grid_size, dim=0)
        s_rep[:, j] = reports_grid.repeat(batch)
        out = mechanism_forward(model, b_rep, s_rep, public_state)
        _, u_j = utilities(
            out,
            buyer_reports.repeat_interleave(grid_size, dim=0),
            seller_reports.repeat_interleave(grid_size, dim=0),
        )
        best_j = torch.maximum(truthful_seller_u[:, j], u_j[:, j].reshape(batch, grid_size).max(dim=1).values)
        seller_regrets.append(torch.relu(best_j - truthful_seller_u[:, j]))
    return torch.stack(buyer_regrets, dim=1), torch.stack(seller_regrets, dim=1)


def simulate_dynamic(
    model: nn.Module,
    cfg: DynamicConfig,
    batch_size: int,
    train: bool,
    value_net: ContinuationValueNet | None = None,
) -> Dict[str, torch.Tensor]:
    device = torch.device(cfg.device)
    buyer_values = torch.zeros(batch_size, cfg.max_buyers, device=device)
    seller_costs = torch.ones(batch_size, cfg.max_sellers, device=device)
    buyer_active = torch.zeros_like(buyer_values)
    seller_active = torch.zeros_like(seller_costs)
    buyer_ages = torch.zeros_like(buyer_values)
    seller_ages = torch.zeros_like(seller_costs)

    rewards = []
    raw_rewards = []
    surpluses = []
    wait_costs = []
    volumes = []
    abandonments = []
    buyer_regrets = []
    seller_regrets = []
    active_regret_samples = []
    queue_lengths = []
    state_values = []
    efficient_volumes = []
    efficient_surpluses = []
    congestion_pressures = []
    congestion_aux_losses = []
    imitation_aux_losses = []

    for t in range(cfg.horizon):
        buyer_values, buyer_active, buyer_ages = compact_queue(buyer_values, buyer_active, buyer_ages, 0.0)
        seller_costs, seller_active, seller_ages = compact_queue(seller_costs, seller_active, seller_ages, 1.0)
        buyer_values, buyer_active, buyer_ages = add_arrivals(buyer_values, buyer_active, buyer_ages, cfg.arrival_prob_buyer, 0.0, True)
        seller_costs, seller_active, seller_ages = add_arrivals(seller_costs, seller_active, seller_ages, cfg.arrival_prob_seller, 1.0, False)

        public_state = public_queue_features(buyer_active, seller_active, buyer_ages, seller_ages, t, cfg)
        if value_net is not None:
            value_state = state_features(buyer_values, seller_costs, buyer_active, seller_active, buyer_ages, seller_ages, t, cfg)
            state_values.append(value_net(value_state))

        buyer_reports, seller_reports = make_reports(buyer_values, seller_costs, buyer_active, seller_active)
        out = mechanism_forward(model, buyer_reports, seller_reports, public_state)
        active_pair = buyer_active[:, :, None] * seller_active[:, None, :]
        match = out["match"] * active_pair
        pair_surplus = buyer_values[:, :, None] - seller_costs[:, None, :]
        surplus = (match * pair_surplus).sum(dim=(1, 2))
        volume = match.sum(dim=(1, 2))
        efficient_volume, efficient_surplus = efficient_queue_targets(buyer_values, seller_costs, buyer_active, seller_active)
        congestion_pressure = queue_congestion_pressure(public_state)
        volume_gap = torch.relu(efficient_volume - volume) / max(min(cfg.max_buyers, cfg.max_sellers), 1)
        surplus_gap = torch.relu(efficient_surplus - surplus)
        volume_weight = min(max(cfg.congestion_volume_weight, 0.0), 1.0)
        congestion_aux_loss = congestion_pressure * (volume_weight * volume_gap + (1.0 - volume_weight) * surplus_gap)
        imitation_aux_loss = torch.zeros_like(congestion_aux_loss)
        if cfg.imitation_aux_weight > 0.0:
            target_match = queue_aware_imitation_match(buyer_reports, seller_reports, public_state, cfg) * active_pair
            imitation_aux_loss = congestion_pressure * (match - target_match).pow(2).sum(dim=(1, 2)) / max(min(cfg.max_buyers, cfg.max_sellers), 1)

        buyer_alloc = torch.clamp(match.sum(dim=2), 0.0, 1.0)
        seller_alloc = torch.clamp(match.sum(dim=1), 0.0, 1.0)
        unmatched_buyers = buyer_active * (1.0 - buyer_alloc)
        unmatched_sellers = seller_active * (1.0 - seller_alloc)
        wait_penalty = cfg.wait_cost * (unmatched_buyers.sum(dim=1) + unmatched_sellers.sum(dim=1))
        reward = surplus - wait_penalty
        discount = cfg.discount**t
        rewards.append(discount * reward)
        raw_rewards.append(reward)
        surpluses.append(surplus)
        wait_costs.append(wait_penalty)
        volumes.append(volume)
        queue_lengths.append(buyer_active.sum(dim=1) + seller_active.sum(dim=1))
        efficient_volumes.append(efficient_volume)
        efficient_surpluses.append(efficient_surplus)
        congestion_pressures.append(congestion_pressure)
        congestion_aux_losses.append(congestion_aux_loss)
        imitation_aux_losses.append(imitation_aux_loss)

        if train:
            b_reg, s_reg = dynamic_regret(model, buyer_reports, seller_reports, cfg, public_state)
            active_b_reg = b_reg * buyer_active
            active_s_reg = s_reg * seller_active
            buyer_regrets.append(active_b_reg.sum() / torch.clamp(buyer_active.sum(), min=1.0))
            seller_regrets.append(active_s_reg.sum() / torch.clamp(seller_active.sum(), min=1.0))
            if torch.any(buyer_active > 0.5):
                active_regret_samples.append(active_b_reg[buyer_active > 0.5])
            if torch.any(seller_active > 0.5):
                active_regret_samples.append(active_s_reg[seller_active > 0.5])

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
        abandonments.append(buyer_abandon.float().sum(dim=1) + seller_abandon.float().sum(dim=1))

        buyer_active = torch.where(depart_buyers | buyer_abandon, torch.zeros_like(buyer_active), buyer_active)
        seller_active = torch.where(depart_sellers | seller_abandon, torch.zeros_like(seller_active), seller_active)
        buyer_values = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
        seller_costs = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
        buyer_ages = torch.where(buyer_active > 0.5, buyer_ages, torch.zeros_like(buyer_ages))
        seller_ages = torch.where(seller_active > 0.5, seller_ages, torch.zeros_like(seller_ages))

    total_reward = torch.stack(rewards, dim=0).sum(dim=0)
    mean_regret = torch.tensor(0.0, device=device)
    p95_regret = torch.tensor(0.0, device=device)
    max_regret = torch.tensor(0.0, device=device)
    active_regret_count = torch.tensor(0.0, device=device)
    if buyer_regrets:
        mean_regret = torch.stack(buyer_regrets).mean() + torch.stack(seller_regrets).mean()
    if active_regret_samples:
        all_active_regrets = torch.cat([values.reshape(-1) for values in active_regret_samples])
        if all_active_regrets.numel() > 0:
            p95_regret = torch.quantile(all_active_regrets, 0.95)
            max_regret = all_active_regrets.max()
            active_regret_count = torch.tensor(float(all_active_regrets.numel()), device=device)

    reward_series = torch.stack(rewards, dim=0)
    continuation = torch.flip(torch.cumsum(torch.flip(reward_series.detach(), dims=[0]), dim=0), dims=[0])
    bellman_residual = torch.tensor(0.0, device=device)
    pathwise_bellman_residual = torch.tensor(0.0, device=device)
    value_loss = torch.tensor(0.0, device=device)
    value_mae = torch.tensor(0.0, device=device)
    value_rmse = torch.tensor(0.0, device=device)
    if cfg.horizon > 1:
        lhs = continuation[:-1]
        rhs = reward_series[:-1].detach() + cfg.discount * continuation[1:]
        bellman_residual = (lhs - rhs).abs().mean()
    if value_net is not None and state_values:
        values = torch.stack(state_values, dim=0)
        rewards_raw = torch.stack(raw_rewards, dim=0).detach()
        returns = []
        running = torch.zeros_like(rewards_raw[-1])
        for reward_t in torch.flip(rewards_raw, dims=[0]):
            running = reward_t + cfg.discount * running
            returns.append(running)
        mc_returns = torch.flip(torch.stack(returns, dim=0), dims=[0])
        next_values = torch.cat([values[1:].detach(), torch.zeros_like(values[-1:])], dim=0)
        td_target = rewards_raw + cfg.discount * next_values
        td_error = values - td_target
        value_error = values - mc_returns.detach()
        value_loss = value_error.pow(2).mean()
        value_mae = value_error.abs().mean()
        value_rmse = torch.sqrt(value_loss + 1.0e-12)
        pathwise_bellman_residual = td_error.abs().mean()
        bellman_residual = td_error.mean(dim=1).abs().mean()

    return {
        "objective": total_reward.mean(),
        "mean_surplus": torch.stack(surpluses).mean(),
        "mean_wait_cost": torch.stack(wait_costs).mean(),
        "mean_volume": torch.stack(volumes).mean(),
        "mean_efficient_volume": torch.stack(efficient_volumes).mean(),
        "mean_efficient_surplus": torch.stack(efficient_surpluses).mean(),
        "mean_congestion_pressure": torch.stack(congestion_pressures).mean(),
        "congestion_aux_loss": torch.stack(congestion_aux_losses).mean(),
        "imitation_aux_loss": torch.stack(imitation_aux_losses).mean(),
        "mean_queue_length": torch.stack(queue_lengths).mean(),
        "mean_abandonment": torch.stack(abandonments).mean(),
        "mean_regret": mean_regret,
        "p95_regret": p95_regret,
        "max_regret": max_regret,
        "active_regret_count": active_regret_count,
        "bellman_residual": bellman_residual,
        "pathwise_bellman_residual": pathwise_bellman_residual,
        "value_loss": value_loss,
        "value_mae": value_mae,
        "value_rmse": value_rmse,
    }


def evaluate_static_baselines(cfg: DynamicConfig) -> Dict[str, float]:
    buyers = np.random.rand(cfg.eval_episodes, cfg.max_buyers)
    sellers = np.random.rand(cfg.eval_episodes, cfg.max_sellers)
    fb = np.maximum(-np.sort(-buyers, axis=1) - np.sort(sellers, axis=1), 0.0).sum(axis=1)
    denom = np.maximum(fb, 1.0e-9)
    tr = trade_reduction_welfare_np(buyers, sellers)
    mc = mcafee_welfare_np(buyers, sellers)
    pp = posted_price_welfare_np(buyers, sellers)
    return {
        "static_trade_reduction_efficiency": float((tr / denom).mean()),
        "static_mcafee_efficiency": float((mc / denom).mean()),
        "static_posted_price_efficiency": float((pp / denom).mean()),
    }


def dynamic_trade_count(buyers: list[float], sellers: list[float], policy: str) -> int:
    b_sorted = sorted(buyers, reverse=True)
    s_sorted = sorted(sellers)
    kmax = min(len(b_sorted), len(s_sorted))
    if kmax == 0:
        return 0
    surplus = [b_sorted[k] - s_sorted[k] for k in range(kmax)]
    efficient_k = sum(x >= 0 for x in surplus)
    if policy == "first_best":
        return efficient_k
    if policy == "trade_reduction":
        return max(efficient_k - 1, 0)
    if policy == "posted":
        return min(sum(b >= 0.5 for b in b_sorted), sum(s <= 0.5 for s in s_sorted))
    if policy == "mcafee":
        if efficient_k == 0:
            return 0
        trade_k = efficient_k - 1
        if efficient_k < len(b_sorted) and efficient_k < len(s_sorted):
            price = 0.5 * (b_sorted[efficient_k] + s_sorted[efficient_k])
            if s_sorted[efficient_k - 1] <= price <= b_sorted[efficient_k - 1]:
                trade_k = efficient_k
        return max(trade_k, 0)
    if policy == "no_trade":
        return 0
    raise ValueError(policy)


StatePostedParams = Tuple[float, float, float, float]


def state_posted_price(
    buyers: list[tuple[float, int]],
    sellers: list[tuple[float, int]],
    t: int,
    cfg: DynamicConfig,
    params: StatePostedParams,
) -> float:
    """Posted price based only on public queue state, not reported values."""
    base, imbalance_weight, age_weight, time_weight = params
    total_capacity = max(cfg.max_buyers + cfg.max_sellers, 1)
    buyer_count = len(buyers)
    seller_count = len(sellers)
    mean_b_age = float(np.mean([age for _, age in buyers])) if buyers else 0.0
    mean_s_age = float(np.mean([age for _, age in sellers])) if sellers else 0.0
    imbalance = (buyer_count - seller_count) / total_capacity
    age_gap = (mean_b_age - mean_s_age) / max(cfg.max_patience, 1)
    time_to_go = 1.0 - float(t) / max(cfg.horizon - 1, 1)
    price = base + imbalance_weight * imbalance + age_weight * age_gap + time_weight * (time_to_go - 0.5)
    return float(np.clip(price, 0.05, 0.95))


def state_posted_trade_count(
    buyers: list[tuple[float, int]],
    sellers: list[tuple[float, int]],
    t: int,
    cfg: DynamicConfig,
    params: StatePostedParams,
) -> int:
    price = state_posted_price(buyers, sellers, t, cfg, params)
    return min(sum(value >= price for value, _ in buyers), sum(cost <= price for cost, _ in sellers))


def queue_aware_mcafee_trade_count(
    buyers: list[tuple[float, int]],
    sellers: list[tuple[float, int]],
    t: int,
    cfg: DynamicConfig,
) -> int:
    """McAfee-style rank rule that relaxes trade reduction in congested states."""
    b_values = [value for value, _ in buyers]
    s_values = [cost for cost, _ in sellers]
    efficient_k = dynamic_trade_count(b_values, s_values, "first_best")
    mcafee_k = dynamic_trade_count(b_values, s_values, "mcafee")
    if efficient_k <= mcafee_k:
        return mcafee_k
    mean_age = 0.0
    if buyers or sellers:
        mean_age = float(np.mean([age for _, age in buyers + sellers]))
    imbalance = abs(len(buyers) - len(sellers)) / max(cfg.max_buyers + cfg.max_sellers, 1)
    t_frac = float(t) / max(cfg.horizon - 1, 1)
    late_market = t_frac >= terminal_pressure_cutoff(cfg)
    congested = (
        mean_age >= cfg.queue_age_trigger * cfg.max_patience
        or imbalance >= cfg.queue_imbalance_trigger
        or late_market
    )
    return efficient_k if congested else mcafee_k


def simulate_dynamic_baseline(
    cfg: DynamicConfig,
    policy: str,
    seed_offset: int = 10_000,
    state_posted_params: StatePostedParams | None = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    objectives = []
    surpluses = []
    waits = []
    volumes = []
    queues = []
    abandons = []
    for _ in range(cfg.eval_episodes):
        buyers: list[tuple[float, int]] = []
        sellers: list[tuple[float, int]] = []
        total_obj = 0.0
        for t in range(cfg.horizon):
            if len(buyers) < cfg.max_buyers and rng.random() <= cfg.arrival_prob_buyer:
                buyers.append((float(rng.random()), 0))
            if len(sellers) < cfg.max_sellers and rng.random() <= cfg.arrival_prob_seller:
                sellers.append((float(rng.random()), 0))
            b_sorted = sorted(buyers, key=lambda x: x[0], reverse=True)
            s_sorted = sorted(sellers, key=lambda x: x[0])
            if policy == "state_posted":
                params = state_posted_params if state_posted_params is not None else (0.5, 0.0, 0.0, 0.0)
                trade_k = state_posted_trade_count(b_sorted, s_sorted, t, cfg, params)
            elif policy == "queue_mcafee":
                trade_k = queue_aware_mcafee_trade_count(b_sorted, s_sorted, t, cfg)
            else:
                trade_k = dynamic_trade_count([b for b, _ in b_sorted], [s for s, _ in s_sorted], policy)
            period_surplus = 0.0
            if trade_k > 0:
                period_surplus = float(sum(b for b, _ in b_sorted[:trade_k]) - sum(s for s, _ in s_sorted[:trade_k]))
            remaining_buyers = b_sorted[trade_k:]
            remaining_sellers = s_sorted[trade_k:]
            wait_cost = cfg.wait_cost * (len(remaining_buyers) + len(remaining_sellers))
            total_obj += (cfg.discount**t) * (period_surplus - wait_cost)
            surpluses.append(period_surplus)
            waits.append(wait_cost)
            volumes.append(float(trade_k))
            queues.append(float(len(buyers) + len(sellers)))
            next_buyers: list[tuple[float, int]] = []
            next_sellers: list[tuple[float, int]] = []
            abandoned = 0
            for value, age in remaining_buyers:
                new_age = age + 1
                prob = min(cfg.abandon_base + cfg.abandon_slope * new_age / max(cfg.max_patience, 1), 1.0)
                if new_age >= cfg.max_patience or rng.random() < prob:
                    abandoned += 1
                else:
                    next_buyers.append((value, new_age))
            for cost, age in remaining_sellers:
                new_age = age + 1
                prob = min(cfg.abandon_base + cfg.abandon_slope * new_age / max(cfg.max_patience, 1), 1.0)
                if new_age >= cfg.max_patience or rng.random() < prob:
                    abandoned += 1
                else:
                    next_sellers.append((cost, new_age))
            abandons.append(float(abandoned))
            buyers = next_buyers
            sellers = next_sellers
        objectives.append(total_obj)
    return {
        f"dynamic_{policy}_objective": float(np.mean(objectives)),
        f"dynamic_{policy}_surplus": float(np.mean(surpluses)),
        f"dynamic_{policy}_wait_cost": float(np.mean(waits)),
        f"dynamic_{policy}_volume": float(np.mean(volumes)),
        f"dynamic_{policy}_queue_length": float(np.mean(queues)),
        f"dynamic_{policy}_abandonment": float(np.mean(abandons)),
    }


def optimize_state_posted_baseline(cfg: DynamicConfig) -> Tuple[StatePostedParams, Dict[str, float]]:
    """Oracle grid search over queue-state posted-price schedules."""
    bases = [0.40, 0.45, 0.50, 0.55, 0.60]
    imbalance_weights = [0.0, 0.20, 0.35, 0.50]
    age_weights = [0.0, 0.10, 0.20, 0.35]
    time_weights = [-0.10, 0.0, 0.10]
    best_params: StatePostedParams = (0.5, 0.0, 0.0, 0.0)
    best_score = -float("inf")
    best_metrics: Dict[str, float] = {}
    for base in bases:
        for imbalance_weight in imbalance_weights:
            for age_weight in age_weights:
                for time_weight in time_weights:
                    params = (base, imbalance_weight, age_weight, time_weight)
                    metrics = simulate_dynamic_baseline(
                        cfg,
                        "state_posted",
                        seed_offset=10_000,
                        state_posted_params=params,
                    )
                    score = metrics["dynamic_state_posted_objective"]
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
    final_metrics = dict(best_metrics)
    base, imbalance_weight, age_weight, time_weight = best_params
    final_metrics.update(
        {
            "dynamic_state_posted_base": base,
            "dynamic_state_posted_imbalance_weight": imbalance_weight,
            "dynamic_state_posted_age_weight": age_weight,
            "dynamic_state_posted_time_weight": time_weight,
            "dynamic_state_posted_search_episodes": float(cfg.eval_episodes),
            "dynamic_state_posted_search_objective": float(best_score),
        }
    )
    return best_params, final_metrics


def evaluate_dynamic_baselines(cfg: DynamicConfig) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for policy in ["first_best", "mcafee", "posted", "trade_reduction", "no_trade"]:
        metrics.update(simulate_dynamic_baseline(cfg, policy))
    _, state_posted_metrics = optimize_state_posted_baseline(cfg)
    metrics.update(state_posted_metrics)
    metrics.update(simulate_dynamic_baseline(cfg, "queue_mcafee", seed_offset=10_000))
    first_best = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    for policy in ["mcafee", "posted", "trade_reduction", "no_trade", "state_posted", "queue_mcafee"]:
        metrics[f"dynamic_{policy}_efficiency"] = metrics[f"dynamic_{policy}_objective"] / first_best
    return metrics


def train_dynamic(cfg: DynamicConfig) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_dynamic_model(cfg).to(device)
    value_net = ContinuationValueNet(hidden=cfg.value_hidden).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(value_net.parameters()), lr=cfg.lr, weight_decay=1.0e-4)
    lam = cfg.regret_weight
    rho = cfg.augmented_rho
    history = []
    best_score = -float("inf")
    best_step = 0
    best_model_state = None
    best_value_state = None

    for step in range(1, cfg.train_steps + 1):
        sim = simulate_dynamic(model, cfg, cfg.batch_size, train=True, value_net=value_net)
        regret_gap = sim["mean_regret"] - cfg.regret_target
        violation = torch.relu(regret_gap)
        loss = (
            -sim["objective"]
            + lam * regret_gap
            + 0.5 * rho * violation.pow(2)
            + cfg.value_loss_weight * sim["value_loss"]
            + cfg.congestion_aux_weight * sim["congestion_aux_loss"]
            + cfg.imitation_aux_weight * sim["imitation_aux_loss"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        gap_float = float(regret_gap.detach().item())
        lam = min(max(lam + rho * gap_float, 0.0), 100.0)
        if step % cfg.log_every == 0 and gap_float > 0.0:
            rho = min(rho * 1.15, 100.0)

        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "objective": float(sim["objective"].item()),
                "mean_surplus": float(sim["mean_surplus"].item()),
                "mean_wait_cost": float(sim["mean_wait_cost"].item()),
                "mean_volume": float(sim["mean_volume"].item()),
                "mean_efficient_volume": float(sim["mean_efficient_volume"].item()),
                "mean_congestion_pressure": float(sim["mean_congestion_pressure"].item()),
                "congestion_aux_loss": float(sim["congestion_aux_loss"].item()),
                "imitation_aux_loss": float(sim["imitation_aux_loss"].item()),
                "mean_queue_length": float(sim["mean_queue_length"].item()),
                "mean_abandonment": float(sim["mean_abandonment"].item()),
                "mean_regret": float(sim["mean_regret"].item()),
                "bellman_residual": float(sim["bellman_residual"].item()),
                "value_loss": float(sim["value_loss"].item()),
                "lambda": float(lam),
                "rho": float(rho),
            }
            history.append(row)
            if cfg.select_best and step >= cfg.selection_min_step:
                selection_score = row["objective"] - cfg.selection_regret_penalty * max(row["mean_regret"] - cfg.regret_target, 0.0)
                row["selection_score"] = float(selection_score)
                if selection_score > best_score:
                    best_score = float(selection_score)
                    best_step = step
                    best_model_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                    best_value_state = copy.deepcopy({key: value.detach().cpu() for key, value in value_net.state_dict().items()})
            print(
                f"step={step:04d} obj={row['objective']:.4f} surplus={row['mean_surplus']:.4f} "
                f"wait={row['mean_wait_cost']:.4f} regret={row['mean_regret']:.4f} "
                f"queue={row['mean_queue_length']:.2f} clear_gap={row['congestion_aux_loss']:.4f} "
                f"imit={row['imitation_aux_loss']:.4f} "
                f"abandon={row['mean_abandonment']:.3f} "
                f"td={row['bellman_residual']:.4f} lambda={lam:.2f}",
                flush=True,
            )

    if cfg.select_best and best_model_state is not None and best_value_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_model_state.items()})
        value_net.load_state_dict({key: value.to(device) for key, value in best_value_state.items()})
        print(f"selected_checkpoint step={best_step:04d} score={best_score:.4f}", flush=True)

    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "training_history.csv", index=False)
    refine_hist = refine_value_net(model, value_net, cfg, out_dir)
    torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save(value_net.state_dict(), out_dir / "value_model.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    model.eval()
    value_net.eval()
    with torch.no_grad():
        eval_sim = simulate_dynamic(model, cfg, cfg.eval_episodes, train=True, value_net=value_net)
    metrics = {key: float(value.item()) for key, value in eval_sim.items()}
    metrics.update(evaluate_static_baselines(cfg))
    metrics.update(evaluate_dynamic_baselines(cfg))
    first_best_obj = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    metrics["dynamic_neural_efficiency"] = metrics["objective"] / first_best_obj
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_history(hist, out_dir)
    return model, hist, metrics


def refine_value_net(
    model: nn.Module,
    value_net: ContinuationValueNet,
    cfg: DynamicConfig,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Policy-evaluation phase for the continuation-value network.

    The mechanism is frozen. Only the value network is trained, using Monte
    Carlo continuation returns plus a one-step Bellman consistency penalty.
    """
    if cfg.value_refine_steps <= 0:
        return pd.DataFrame()

    was_training = model.training
    model.eval()
    previous_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    value_net.train()
    opt = torch.optim.AdamW(value_net.parameters(), lr=cfg.value_refine_lr, weight_decay=1.0e-4)
    rows = []
    for step in range(1, cfg.value_refine_steps + 1):
        sim = simulate_dynamic(model, cfg, cfg.value_refine_batch_size, train=False, value_net=value_net)
        loss = sim["value_loss"] + cfg.value_refine_td_weight * sim["bellman_residual"].pow(2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 5.0)
        opt.step()
        if step == 1 or step % cfg.value_refine_log_every == 0 or step == cfg.value_refine_steps:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "value_loss": float(sim["value_loss"].item()),
                "bellman_residual": float(sim["bellman_residual"].item()),
                "objective": float(sim["objective"].item()),
            }
            rows.append(row)
            print(
                f"value_refine={step:04d} value_loss={row['value_loss']:.5f} "
                f"td={row['bellman_residual']:.5f} obj={row['objective']:.4f}",
                flush=True,
            )

    for param, requires_grad in zip(model.parameters(), previous_requires_grad):
        param.requires_grad_(requires_grad)
    if was_training:
        model.train()
    else:
        model.eval()
    value_net.eval()

    hist = pd.DataFrame(rows)
    if out_dir is not None and not hist.empty:
        hist.to_csv(out_dir / "value_refine_history.csv", index=False)
        plot_value_refinement(hist, out_dir)
    return hist


def plot_value_refinement(history: pd.DataFrame, out_dir: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), dpi=150)
    axes[0].plot(history["step"], history["value_loss"], color="#0f766e")
    axes[0].set_title("Value prediction loss")
    axes[1].plot(history["step"], history["bellman_residual"], color="#334155")
    axes[1].set_title("Bellman residual")
    for ax in axes:
        ax.set_xlabel("refinement step")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "value_refine_curves.png")
    plt.close(fig)


def plot_history(history: pd.DataFrame, out_dir: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes = axes.ravel()
    axes[0].plot(history["step"], history["objective"], color="#0f766e")
    axes[0].set_title("Discounted dynamic objective")
    axes[1].plot(history["step"], history["mean_regret"], color="#b91c1c")
    axes[1].axhline(history["mean_regret"].iloc[-1], color="#999999", linewidth=0.6)
    axes[1].set_title("Per-period grid regret")
    axes[2].plot(history["step"], history["mean_wait_cost"], color="#6d28d9")
    axes[2].set_title("Waiting cost")
    axes[3].plot(history["step"], history["bellman_residual"], color="#334155", label="TD residual")
    axes[3].plot(history["step"], history["mean_abandonment"], color="#ea580c", label="abandonment")
    axes[3].set_title("Continuation diagnostics")
    axes[3].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_training_curves.png")
    plt.close(fig)


def parse_args() -> DynamicConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-buyers", type=int, default=DynamicConfig.max_buyers)
    parser.add_argument("--max-sellers", type=int, default=DynamicConfig.max_sellers)
    parser.add_argument("--horizon", type=int, default=DynamicConfig.horizon)
    parser.add_argument("--batch-size", type=int, default=DynamicConfig.batch_size)
    parser.add_argument("--train-steps", type=int, default=DynamicConfig.train_steps)
    parser.add_argument("--lr", type=float, default=DynamicConfig.lr)
    parser.add_argument("--hidden", type=int, default=DynamicConfig.hidden)
    parser.add_argument("--depth", type=int, default=DynamicConfig.depth)
    parser.add_argument("--feature-mode", choices=["basic", "ranked"], default=DynamicConfig.feature_mode)
    parser.add_argument("--mechanism", choices=["base", "state"], default=DynamicConfig.mechanism)
    parser.add_argument("--arrival-prob-buyer", type=float, default=DynamicConfig.arrival_prob_buyer)
    parser.add_argument("--arrival-prob-seller", type=float, default=DynamicConfig.arrival_prob_seller)
    parser.add_argument("--wait-cost", type=float, default=DynamicConfig.wait_cost)
    parser.add_argument("--discount", type=float, default=DynamicConfig.discount)
    parser.add_argument("--max-patience", type=int, default=DynamicConfig.max_patience)
    parser.add_argument("--abandon-base", type=float, default=DynamicConfig.abandon_base)
    parser.add_argument("--abandon-slope", type=float, default=DynamicConfig.abandon_slope)
    parser.add_argument("--regret-grid", type=int, default=DynamicConfig.regret_grid)
    parser.add_argument("--regret-method", choices=["grid", "adv", "hybrid"], default=DynamicConfig.regret_method)
    parser.add_argument("--adv-steps", type=int, default=DynamicConfig.adv_steps)
    parser.add_argument("--adv-lr", type=float, default=DynamicConfig.adv_lr)
    parser.add_argument("--adv-restarts", type=int, default=DynamicConfig.adv_restarts)
    parser.add_argument("--regret-weight", type=float, default=DynamicConfig.regret_weight)
    parser.add_argument("--regret-target", type=float, default=DynamicConfig.regret_target)
    parser.add_argument("--augmented-rho", type=float, default=DynamicConfig.augmented_rho)
    parser.add_argument("--congestion-aux-weight", type=float, default=DynamicConfig.congestion_aux_weight)
    parser.add_argument("--congestion-volume-weight", type=float, default=DynamicConfig.congestion_volume_weight)
    parser.add_argument("--imitation-aux-weight", type=float, default=DynamicConfig.imitation_aux_weight)
    parser.add_argument("--value-hidden", type=int, default=DynamicConfig.value_hidden)
    parser.add_argument("--value-loss-weight", type=float, default=DynamicConfig.value_loss_weight)
    parser.add_argument("--value-refine-steps", type=int, default=DynamicConfig.value_refine_steps)
    parser.add_argument("--value-refine-batch-size", type=int, default=DynamicConfig.value_refine_batch_size)
    parser.add_argument("--value-refine-lr", type=float, default=DynamicConfig.value_refine_lr)
    parser.add_argument("--value-refine-td-weight", type=float, default=DynamicConfig.value_refine_td_weight)
    parser.add_argument("--value-refine-log-every", type=int, default=DynamicConfig.value_refine_log_every)
    parser.add_argument("--queue-age-trigger", type=float, default=DynamicConfig.queue_age_trigger)
    parser.add_argument("--queue-imbalance-trigger", type=float, default=DynamicConfig.queue_imbalance_trigger)
    parser.add_argument("--queue-terminal-window", type=int, default=DynamicConfig.queue_terminal_window)
    parser.add_argument("--eval-episodes", type=int, default=DynamicConfig.eval_episodes)
    parser.add_argument("--select-best", action="store_true", default=DynamicConfig.select_best)
    parser.add_argument("--selection-regret-penalty", type=float, default=DynamicConfig.selection_regret_penalty)
    parser.add_argument("--selection-min-step", type=int, default=DynamicConfig.selection_min_step)
    parser.add_argument("--seed", type=int, default=DynamicConfig.seed)
    parser.add_argument("--log-every", type=int, default=DynamicConfig.log_every)
    parser.add_argument("--out-dir", type=str, default=DynamicConfig.out_dir)
    args = parser.parse_args()
    return DynamicConfig(
        max_buyers=args.max_buyers,
        max_sellers=args.max_sellers,
        horizon=args.horizon,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        lr=args.lr,
        hidden=args.hidden,
        depth=args.depth,
        feature_mode=args.feature_mode,
        mechanism=args.mechanism,
        arrival_prob_buyer=args.arrival_prob_buyer,
        arrival_prob_seller=args.arrival_prob_seller,
        wait_cost=args.wait_cost,
        discount=args.discount,
        max_patience=args.max_patience,
        abandon_base=args.abandon_base,
        abandon_slope=args.abandon_slope,
        regret_grid=args.regret_grid,
        regret_method=args.regret_method,
        adv_steps=args.adv_steps,
        adv_lr=args.adv_lr,
        adv_restarts=args.adv_restarts,
        regret_weight=args.regret_weight,
        regret_target=args.regret_target,
        augmented_rho=args.augmented_rho,
        congestion_aux_weight=args.congestion_aux_weight,
        congestion_volume_weight=args.congestion_volume_weight,
        imitation_aux_weight=args.imitation_aux_weight,
        value_hidden=args.value_hidden,
        value_loss_weight=args.value_loss_weight,
        value_refine_steps=args.value_refine_steps,
        value_refine_batch_size=args.value_refine_batch_size,
        value_refine_lr=args.value_refine_lr,
        value_refine_td_weight=args.value_refine_td_weight,
        value_refine_log_every=args.value_refine_log_every,
        queue_age_trigger=args.queue_age_trigger,
        queue_imbalance_trigger=args.queue_imbalance_trigger,
        queue_terminal_window=args.queue_terminal_window,
        eval_episodes=args.eval_episodes,
        select_best=args.select_best,
        selection_regret_penalty=args.selection_regret_penalty,
        selection_min_step=args.selection_min_step,
        seed=args.seed,
        log_every=args.log_every,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    _, _, metrics = train_dynamic(cfg)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
