from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import torch
from torch import nn


@dataclass
class Config:
    n_buyers: int = 3
    n_sellers: int = 3
    batch_size: int = 384
    train_steps: int = 900
    lr: float = 2.0e-3
    hidden: int = 64
    depth: int = 3
    feature_mode: str = "basic"
    regret_grid: int = 9
    eval_grid: int = 31
    eval_samples: int = 6000
    regret_eval_samples: int = 1200
    regret_weight: float = 8.0
    train_method: str = "penalty"
    regret_method: str = "grid"
    eval_regret_method: str = "grid"
    regret_target: float = 0.006
    dense_grid: int = 101
    dual_lr: float = 0.75
    max_regret_weight: float = 80.0
    adv_steps: int = 8
    adv_lr: float = 0.8
    adv_restarts: int = 3
    exact_maxiter: int = 30
    exact_popsize: int = 8
    exact_tol: float = 1.0e-5
    augmented_rho: float = 8.0
    rho_growth: float = 1.15
    max_augmented_rho: float = 100.0
    distribution: str = "uniform"
    clearance_cost: float = 0.0
    log_every: int = 50
    seed: int = 7
    out_dir: str = "outputs"
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_market(
    batch: int,
    n_buyers: int,
    n_sellers: int,
    device: torch.device,
    distribution: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample independent-private-value double-auction primitives on [0, 1]."""
    if distribution == "uniform":
        buyers = torch.rand(batch, n_buyers, device=device)
        sellers = torch.rand(batch, n_sellers, device=device)
    elif distribution == "beta_easy":
        buyers = torch.distributions.Beta(2.2, 1.4).sample((batch, n_buyers)).to(device)
        sellers = torch.distributions.Beta(1.4, 2.2).sample((batch, n_sellers)).to(device)
    elif distribution == "beta_hard":
        buyers = torch.distributions.Beta(2.0, 2.0).sample((batch, n_buyers)).to(device)
        sellers = torch.distributions.Beta(2.0, 2.0).sample((batch, n_sellers)).to(device)
    elif distribution == "asymmetric":
        buyers = 0.15 + 0.85 * torch.distributions.Beta(1.8, 1.3).sample((batch, n_buyers)).to(device)
        sellers = 0.85 * torch.distributions.Beta(1.3, 1.8).sample((batch, n_sellers)).to(device)
    elif distribution == "correlated":
        factor = torch.rand(batch, 1, device=device)
        buyers = torch.clamp(0.55 * factor + 0.45 * torch.rand(batch, n_buyers, device=device), 0.0, 1.0)
        sellers = torch.clamp(0.55 * factor + 0.45 * torch.rand(batch, n_sellers, device=device), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return buyers, sellers


class HardConstrainedDoubleAuction(nn.Module):
    """
    A permutation-equivariant, pairwise neural direct mechanism.

    Hard constraints:
    - no reported negative-surplus trade;
    - row/column sums of the fractional matching matrix are <= 1;
    - buyer payments are no greater than reported values under truthful reporting;
    - seller transfers are no lower than reported costs under truthful reporting;
    - buyer payments weakly exceed seller transfers, so ex-post budget balance holds.
    """

    def __init__(
        self,
        hidden: int = 64,
        depth: int = 3,
        feature_mode: str = "basic",
        clearance_cost: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        self.clearance_cost = float(clearance_cost)
        layers = []
        width_in = 6 if feature_mode == "basic" else 10
        for layer_idx in range(depth):
            layers.append(nn.Linear(width_in if layer_idx == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, buyer_reports: torch.Tensor, seller_reports: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, n_buyers = buyer_reports.shape
        n_sellers = seller_reports.shape[1]

        b = buyer_reports[:, :, None].expand(batch, n_buyers, n_sellers)
        a = seller_reports[:, None, :].expand(batch, n_buyers, n_sellers)
        spread = b - a
        mean_b = buyer_reports.mean(dim=1, keepdim=True)[:, :, None].expand_as(b)
        mean_a = seller_reports.mean(dim=1, keepdim=True)[:, None, :].expand_as(a)
        market_tightness = mean_b - mean_a
        feasible = (spread >= self.clearance_cost).float()

        features = torch.stack(
            self._features(buyer_reports, seller_reports, b, a, spread, mean_b.expand_as(b), mean_a.expand_as(a), market_tightness),
            dim=-1,
        )
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


def utilities(
    outcome: Dict[str, torch.Tensor],
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    buyer_u = buyer_values * outcome["buyer_alloc"] - outcome["buyer_payments"]
    seller_u = outcome["seller_transfers"] - seller_costs * outcome["seller_alloc"]
    return buyer_u, seller_u


def welfare(
    outcome: Dict[str, torch.Tensor],
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    clearance_cost: float = 0.0,
) -> torch.Tensor:
    pair_surplus = buyer_values[:, :, None] - seller_costs[:, None, :]
    net_surplus = pair_surplus - float(clearance_cost)
    return (outcome["match"] * net_surplus).sum(dim=(1, 2))


def grid_regret(
    model: HardConstrainedDoubleAuction,
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, n_buyers = buyer_values.shape
    n_sellers = seller_costs.shape[1]
    device = buyer_values.device
    reports_grid = torch.linspace(0.0, 1.0, grid_size, device=device)

    truthful_outcome = model(buyer_values, seller_costs)
    truthful_buyer_u, truthful_seller_u = utilities(truthful_outcome, buyer_values, seller_costs)

    buyer_regrets = []
    for i in range(n_buyers):
        b_rep = buyer_values.repeat_interleave(grid_size, dim=0)
        s_rep = seller_costs.repeat_interleave(grid_size, dim=0)
        b_rep[:, i] = reports_grid.repeat(batch)
        out = model(b_rep, s_rep)
        u_i, _ = utilities(out, buyer_values.repeat_interleave(grid_size, dim=0), seller_costs.repeat_interleave(grid_size, dim=0))
        best = u_i[:, i].view(batch, grid_size).max(dim=1).values
        buyer_regrets.append(torch.relu(best - truthful_buyer_u[:, i]))

    seller_regrets = []
    for j in range(n_sellers):
        b_rep = buyer_values.repeat_interleave(grid_size, dim=0)
        s_rep = seller_costs.repeat_interleave(grid_size, dim=0)
        s_rep[:, j] = reports_grid.repeat(batch)
        out = model(b_rep, s_rep)
        _, u_j = utilities(out, buyer_values.repeat_interleave(grid_size, dim=0), seller_costs.repeat_interleave(grid_size, dim=0))
        best = u_j[:, j].view(batch, grid_size).max(dim=1).values
        seller_regrets.append(torch.relu(best - truthful_seller_u[:, j]))

    return torch.stack(buyer_regrets, dim=1), torch.stack(seller_regrets, dim=1)


def adversarial_regret(
    model: HardConstrainedDoubleAuction,
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    steps: int,
    lr: float,
    restarts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate ex-post regret with gradient ascent over each agent's report.

    The inner maximization treats misreports as fixed after ascent, so the final
    regret remains differentiable with respect to mechanism parameters. This is
    the usual computational mechanism-design compromise used in RegretNet-style
    training.
    """
    batch, n_buyers = buyer_values.shape
    n_sellers = seller_costs.shape[1]
    device = buyer_values.device

    truthful_outcome = model(buyer_values, seller_costs)
    truthful_buyer_u, truthful_seller_u = utilities(truthful_outcome, buyer_values, seller_costs)
    buyer_base = buyer_values.repeat_interleave(restarts, dim=0)
    seller_base = seller_costs.repeat_interleave(restarts, dim=0)

    buyer_regrets = []
    for i in range(n_buyers):
        z = torch.randn(batch * restarts, device=device, requires_grad=True)
        for _ in range(steps):
            misreport = torch.sigmoid(z)
            b_rep = buyer_base.clone()
            b_rep[:, i] = misreport
            out = model(b_rep, seller_base)
            u_i, _ = utilities(out, buyer_base, seller_base)
            objective = u_i[:, i].mean()
            grad = torch.autograd.grad(objective, z, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                z.add_(lr * grad.sign())
            z.requires_grad_(True)

        final_misreport = torch.sigmoid(z.detach())
        b_rep = buyer_base.clone()
        b_rep[:, i] = final_misreport
        out = model(b_rep, seller_base)
        u_i, _ = utilities(out, buyer_base, seller_base)
        best = u_i[:, i].view(batch, restarts).max(dim=1).values
        buyer_regrets.append(torch.relu(best - truthful_buyer_u[:, i]))

    seller_regrets = []
    for j in range(n_sellers):
        z = torch.randn(batch * restarts, device=device, requires_grad=True)
        for _ in range(steps):
            misreport = torch.sigmoid(z)
            s_rep = seller_base.clone()
            s_rep[:, j] = misreport
            out = model(buyer_base, s_rep)
            _, u_j = utilities(out, buyer_base, seller_base)
            objective = u_j[:, j].mean()
            grad = torch.autograd.grad(objective, z, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                z.add_(lr * grad.sign())
            z.requires_grad_(True)

        final_misreport = torch.sigmoid(z.detach())
        s_rep = seller_base.clone()
        s_rep[:, j] = final_misreport
        out = model(buyer_base, s_rep)
        _, u_j = utilities(out, buyer_base, seller_base)
        best = u_j[:, j].view(batch, restarts).max(dim=1).values
        seller_regrets.append(torch.relu(best - truthful_seller_u[:, j]))

    return torch.stack(buyer_regrets, dim=1), torch.stack(seller_regrets, dim=1)


def exact_best_response_regret(
    model: HardConstrainedDoubleAuction,
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    maxiter: int,
    popsize: int,
    tol: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Derivative-free one-dimensional best-response audit.

    This is intended for evaluation, not training. Each agent's report is
    optimized on [0, 1] with differential evolution. The resulting audit is much
    slower than grid or adversarial regret but is a stronger check against
    hidden profitable deviations.
    """
    model.eval()
    device = buyer_values.device
    buyers = buyer_values.detach().cpu().numpy()
    sellers = seller_costs.detach().cpu().numpy()
    batch, n_buyers = buyers.shape
    n_sellers = sellers.shape[1]

    with torch.no_grad():
        truthful_outcome = model(buyer_values, seller_costs)
        truthful_buyer_u, truthful_seller_u = utilities(truthful_outcome, buyer_values, seller_costs)
    truthful_buyer = truthful_buyer_u.detach().cpu().numpy()
    truthful_seller = truthful_seller_u.detach().cpu().numpy()

    buyer_regrets = np.zeros((batch, n_buyers), dtype=np.float32)
    seller_regrets = np.zeros((batch, n_sellers), dtype=np.float32)

    def buyer_utility(sample_idx: int, agent_idx: int, report: float) -> float:
        b_rep = buyers[sample_idx].copy()
        b_rep[agent_idx] = float(np.clip(report, 0.0, 1.0))
        with torch.no_grad():
            b_t = torch.tensor(b_rep[None, :], dtype=buyer_values.dtype, device=device)
            s_t = torch.tensor(sellers[sample_idx][None, :], dtype=seller_costs.dtype, device=device)
            v_t = torch.tensor(buyers[sample_idx][None, :], dtype=buyer_values.dtype, device=device)
            c_t = torch.tensor(sellers[sample_idx][None, :], dtype=seller_costs.dtype, device=device)
            out = model(b_t, s_t)
            u_b, _ = utilities(out, v_t, c_t)
        return float(u_b[0, agent_idx].item())

    def seller_utility(sample_idx: int, agent_idx: int, report: float) -> float:
        s_rep = sellers[sample_idx].copy()
        s_rep[agent_idx] = float(np.clip(report, 0.0, 1.0))
        with torch.no_grad():
            b_t = torch.tensor(buyers[sample_idx][None, :], dtype=buyer_values.dtype, device=device)
            s_t = torch.tensor(s_rep[None, :], dtype=seller_costs.dtype, device=device)
            v_t = torch.tensor(buyers[sample_idx][None, :], dtype=buyer_values.dtype, device=device)
            c_t = torch.tensor(sellers[sample_idx][None, :], dtype=seller_costs.dtype, device=device)
            out = model(b_t, s_t)
            _, u_s = utilities(out, v_t, c_t)
        return float(u_s[0, agent_idx].item())

    for sample_idx in range(batch):
        for i in range(n_buyers):
            result = differential_evolution(
                lambda x: -buyer_utility(sample_idx, i, float(x[0])),
                bounds=[(0.0, 1.0)],
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                polish=True,
                seed=seed + 1009 * sample_idx + 37 * i,
                updating="immediate",
                workers=1,
            )
            buyer_regrets[sample_idx, i] = max(-float(result.fun) - truthful_buyer[sample_idx, i], 0.0)
        for j in range(n_sellers):
            result = differential_evolution(
                lambda x: -seller_utility(sample_idx, j, float(x[0])),
                bounds=[(0.0, 1.0)],
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                polish=True,
                seed=seed + 1009 * sample_idx + 53 * j + 500_000,
                updating="immediate",
                workers=1,
            )
            seller_regrets[sample_idx, j] = max(-float(result.fun) - truthful_seller[sample_idx, j], 0.0)

    return torch.tensor(buyer_regrets, device=device), torch.tensor(seller_regrets, device=device)


def mechanism_regret(
    model: HardConstrainedDoubleAuction,
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    cfg: Config,
    method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "grid":
        return grid_regret(model, buyer_values, seller_costs, cfg.regret_grid)
    if method == "dense":
        return grid_regret(model, buyer_values, seller_costs, cfg.dense_grid)
    if method == "adv":
        return adversarial_regret(model, buyer_values, seller_costs, cfg.adv_steps, cfg.adv_lr, cfg.adv_restarts)
    if method == "hybrid":
        grid_b, grid_s = grid_regret(model, buyer_values, seller_costs, cfg.regret_grid)
        adv_b, adv_s = adversarial_regret(model, buyer_values, seller_costs, cfg.adv_steps, cfg.adv_lr, cfg.adv_restarts)
        return torch.maximum(grid_b, adv_b), torch.maximum(grid_s, adv_s)
    if method in {"exact", "audit"}:
        raise ValueError("exact/audit regret is for evaluation only; use grid, dense, adv, or hybrid for training.")
    raise ValueError(f"Unknown regret method: {method}")


def first_best_welfare_np(buyers: np.ndarray, sellers: np.ndarray, clearance_cost: float = 0.0) -> np.ndarray:
    b_sorted = -np.sort(-buyers, axis=1)
    s_sorted = np.sort(sellers, axis=1)
    kmax = min(buyers.shape[1], sellers.shape[1])
    surplus = b_sorted[:, :kmax] - s_sorted[:, :kmax] - clearance_cost
    return np.maximum(surplus, 0.0).sum(axis=1)


def trade_reduction_welfare_np(buyers: np.ndarray, sellers: np.ndarray, clearance_cost: float = 0.0) -> np.ndarray:
    b_sorted = -np.sort(-buyers, axis=1)
    s_sorted = np.sort(sellers, axis=1)
    kmax = min(buyers.shape[1], sellers.shape[1])
    surplus = b_sorted[:, :kmax] - s_sorted[:, :kmax] - clearance_cost
    efficient_k = (surplus >= 0).sum(axis=1)
    welfare_values = np.zeros(buyers.shape[0], dtype=np.float64)
    for idx, k in enumerate(efficient_k):
        trade_k = max(int(k) - 1, 0)
        if trade_k > 0:
            welfare_values[idx] = surplus[idx, :trade_k].sum()
    return welfare_values


def mcafee_welfare_np(buyers: np.ndarray, sellers: np.ndarray, clearance_cost: float = 0.0) -> np.ndarray:
    b_sorted = -np.sort(-buyers, axis=1)
    s_sorted = np.sort(sellers, axis=1)
    n_buyers = buyers.shape[1]
    n_sellers = sellers.shape[1]
    kmax = min(n_buyers, n_sellers)
    surplus = b_sorted[:, :kmax] - s_sorted[:, :kmax] - clearance_cost
    efficient_k = (surplus >= 0).sum(axis=1)
    welfare_values = np.zeros(buyers.shape[0], dtype=np.float64)
    for idx, k_raw in enumerate(efficient_k):
        k = int(k_raw)
        if k == 0:
            continue
        has_next_buyer = k < n_buyers
        has_next_seller = k < n_sellers
        trade_k = k - 1
        if has_next_buyer and has_next_seller:
            shifted_sellers = s_sorted[idx] + clearance_cost
            candidate_price = 0.5 * (b_sorted[idx, k] + shifted_sellers[k])
            if shifted_sellers[k - 1] <= candidate_price <= b_sorted[idx, k - 1]:
                trade_k = k
        if trade_k > 0:
            welfare_values[idx] = surplus[idx, :trade_k].sum()
    return welfare_values


def posted_price_welfare_np(
    buyers: np.ndarray,
    sellers: np.ndarray,
    price: float = 0.5,
    clearance_cost: float = 0.0,
) -> np.ndarray:
    b_sorted = -np.sort(-buyers, axis=1)
    s_sorted = np.sort(sellers, axis=1)
    buyer_accept = b_sorted >= price
    seller_accept = (s_sorted + clearance_cost) <= price
    q = np.minimum(buyer_accept.sum(axis=1), seller_accept.sum(axis=1))
    welfare_values = np.zeros(buyers.shape[0], dtype=np.float64)
    for idx, trade_k in enumerate(q):
        k = int(trade_k)
        if k > 0:
            welfare_values[idx] = (b_sorted[idx, :k] - s_sorted[idx, :k] - clearance_cost).sum()
    return welfare_values


def evaluate(
    model: HardConstrainedDoubleAuction,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        buyers, sellers = sample_market(cfg.eval_samples, cfg.n_buyers, cfg.n_sellers, device, cfg.distribution)
        out = model(buyers, sellers)
        w = welfare(out, buyers, sellers, cfg.clearance_cost)
        buyer_u, seller_u = utilities(out, buyers, sellers)

        buyer_np = buyers.cpu().numpy()
        seller_np = sellers.cpu().numpy()
        fb = first_best_welfare_np(buyer_np, seller_np, cfg.clearance_cost)
        tr = trade_reduction_welfare_np(buyer_np, seller_np, cfg.clearance_cost)
        mc = mcafee_welfare_np(buyer_np, seller_np, cfg.clearance_cost)
        pp = posted_price_welfare_np(buyer_np, seller_np, clearance_cost=cfg.clearance_cost)
        denom = np.maximum(fb, 1.0e-9)

        budget_surplus = out["buyer_payments"].sum(dim=1) - out["seller_transfers"].sum(dim=1)
        clearing = out["buyer_alloc"].sum(dim=1) - out["seller_alloc"].sum(dim=1)
        row_excess = torch.relu(out["match"].sum(dim=2) - 1.0).max()
        col_excess = torch.relu(out["match"].sum(dim=1) - 1.0).max()

        neural_w = w.cpu().numpy()
        metrics = {
            "neural_welfare": float(neural_w.mean()),
            "first_best_welfare": float(fb.mean()),
            "trade_reduction_welfare": float(tr.mean()),
            "mcafee_welfare": float(mc.mean()),
            "posted_price_welfare": float(pp.mean()),
            "neural_efficiency": float((neural_w / denom).mean()),
            "trade_reduction_efficiency": float((tr / denom).mean()),
            "mcafee_efficiency": float((mc / denom).mean()),
            "posted_price_efficiency": float((pp / denom).mean()),
            "min_budget_surplus": float(budget_surplus.min().item()),
            "mean_budget_surplus": float(budget_surplus.mean().item()),
            "max_clearing_abs": float(clearing.abs().max().item()),
            "max_row_excess": float(row_excess.item()),
            "max_col_excess": float(col_excess.item()),
            "min_buyer_utility": float(buyer_u.min().item()),
            "min_seller_utility": float(seller_u.min().item()),
            "mean_trade_volume": float(out["match"].sum(dim=(1, 2)).mean().item()),
        }

    regret_count = min(cfg.regret_eval_samples, cfg.eval_samples)
    reg_buyers = buyers[:regret_count].detach()
    reg_sellers = sellers[:regret_count].detach()
    if cfg.eval_regret_method == "grid":
        with torch.no_grad():
            b_reg, s_reg = grid_regret(model, reg_buyers, reg_sellers, cfg.eval_grid)
    elif cfg.eval_regret_method == "dense":
        with torch.no_grad():
            b_reg, s_reg = grid_regret(model, reg_buyers, reg_sellers, cfg.dense_grid)
    elif cfg.eval_regret_method == "adv":
        with torch.enable_grad():
            b_reg, s_reg = adversarial_regret(
                model,
                reg_buyers,
                reg_sellers,
                cfg.adv_steps,
                cfg.adv_lr,
                cfg.adv_restarts,
            )
    elif cfg.eval_regret_method == "hybrid":
        with torch.no_grad():
            grid_b, grid_s = grid_regret(model, reg_buyers, reg_sellers, cfg.dense_grid)
        with torch.enable_grad():
            adv_b, adv_s = adversarial_regret(
                model,
                reg_buyers,
                reg_sellers,
                cfg.adv_steps,
                cfg.adv_lr,
                cfg.adv_restarts,
            )
        b_reg, s_reg = torch.maximum(grid_b, adv_b), torch.maximum(grid_s, adv_s)
    elif cfg.eval_regret_method == "exact":
        with torch.no_grad():
            b_reg, s_reg = exact_best_response_regret(
                model,
                reg_buyers,
                reg_sellers,
                cfg.exact_maxiter,
                cfg.exact_popsize,
                cfg.exact_tol,
                cfg.seed,
            )
    elif cfg.eval_regret_method == "audit":
        with torch.no_grad():
            grid_b, grid_s = grid_regret(model, reg_buyers, reg_sellers, cfg.dense_grid)
        with torch.enable_grad():
            adv_b, adv_s = adversarial_regret(
                model,
                reg_buyers,
                reg_sellers,
                cfg.adv_steps,
                cfg.adv_lr,
                cfg.adv_restarts,
            )
        exact_b, exact_s = exact_best_response_regret(
            model,
            reg_buyers,
            reg_sellers,
            cfg.exact_maxiter,
            cfg.exact_popsize,
            cfg.exact_tol,
            cfg.seed,
        )
        b_reg = torch.maximum(torch.maximum(grid_b, adv_b), exact_b)
        s_reg = torch.maximum(torch.maximum(grid_s, adv_s), exact_s)
    else:
        raise ValueError(f"Unknown eval regret method: {cfg.eval_regret_method}")

    all_regrets = torch.cat([b_reg.flatten(), s_reg.flatten()])
    metrics.update(
        {
            "regret_method": cfg.eval_regret_method,
            "regret_eval_samples": float(regret_count),
            "mean_buyer_regret": float(b_reg.mean().item()),
            "max_buyer_regret": float(b_reg.max().item()),
            "p95_buyer_regret": float(torch.quantile(b_reg.flatten(), 0.95).item()),
            "mean_seller_regret": float(s_reg.mean().item()),
            "max_seller_regret": float(s_reg.max().item()),
            "p95_seller_regret": float(torch.quantile(s_reg.flatten(), 0.95).item()),
            "p95_total_agent_regret": float(torch.quantile(all_regrets, 0.95).item()),
        }
    )
    model.train()
    return metrics


def train(cfg: Config) -> Tuple[HardConstrainedDoubleAuction, pd.DataFrame, Dict[str, float]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HardConstrainedDoubleAuction(
        hidden=cfg.hidden,
        depth=cfg.depth,
        feature_mode=cfg.feature_mode,
        clearance_cost=cfg.clearance_cost,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.0e-4)
    history = []
    regret_weight = cfg.regret_weight
    augmented_rho = cfg.augmented_rho

    for step in range(1, cfg.train_steps + 1):
        buyers, sellers = sample_market(cfg.batch_size, cfg.n_buyers, cfg.n_sellers, device, cfg.distribution)
        out = model(buyers, sellers)
        w = welfare(out, buyers, sellers, cfg.clearance_cost)
        b_reg, s_reg = mechanism_regret(model, buyers, sellers, cfg, cfg.regret_method)
        regret = b_reg.mean() + s_reg.mean()
        regret_gap_tensor = regret - cfg.regret_target
        if cfg.train_method == "penalty":
            loss = -w.mean() + regret_weight * regret
        elif cfg.train_method == "dual":
            loss = -w.mean() + regret_weight * regret_gap_tensor
        elif cfg.train_method == "aug_lagrangian":
            violation = torch.relu(regret_gap_tensor)
            loss = -w.mean() + regret_weight * regret_gap_tensor + 0.5 * augmented_rho * violation.pow(2)
        else:
            raise ValueError(f"Unknown train method: {cfg.train_method}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if cfg.train_method == "dual":
            regret_gap = float(regret.detach().item() - cfg.regret_target)
            regret_weight = min(max(regret_weight + cfg.dual_lr * regret_gap, 0.0), cfg.max_regret_weight)
        elif cfg.train_method == "aug_lagrangian":
            regret_gap = float(regret.detach().item() - cfg.regret_target)
            regret_weight = min(max(regret_weight + augmented_rho * regret_gap, 0.0), cfg.max_regret_weight)
            if step % max(cfg.log_every, 1) == 0 and regret_gap > 0.0:
                augmented_rho = min(augmented_rho * cfg.rho_growth, cfg.max_augmented_rho)

        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            budget_surplus = out["buyer_payments"].sum(dim=1) - out["seller_transfers"].sum(dim=1)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "welfare": float(w.mean().item()),
                    "buyer_regret": float(b_reg.mean().item()),
                    "seller_regret": float(s_reg.mean().item()),
                    "total_regret": float(regret.item()),
                    "regret_weight": float(regret_weight),
                    "augmented_rho": float(augmented_rho),
                    "budget_surplus": float(budget_surplus.mean().item()),
                    "trade_volume": float(out["match"].sum(dim=(1, 2)).mean().item()),
                }
            )
            print(
                f"step={step:04d} loss={loss.item():.4f} welfare={w.mean().item():.4f} "
                f"regret=({b_reg.mean().item():.4f},{s_reg.mean().item():.4f}) "
                f"lambda={regret_weight:.3f} rho={augmented_rho:.2f} "
                f"volume={out['match'].sum(dim=(1, 2)).mean().item():.3f}",
                flush=True,
            )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)

    metrics = evaluate(model, cfg, device)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    plot_history(hist_df, out_dir)
    return model, hist_df, metrics


def plot_history(history: pd.DataFrame, out_dir: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes = axes.ravel()
    axes[0].plot(history["step"], history["welfare"], color="#0f766e")
    axes[0].set_title("Training welfare")
    axes[0].set_xlabel("step")
    axes[1].plot(history["step"], history["buyer_regret"], label="buyer", color="#b91c1c")
    axes[1].plot(history["step"], history["seller_regret"], label="seller", color="#1d4ed8")
    axes[1].set_title("IC regret")
    axes[1].set_xlabel("step")
    axes[1].legend(frameon=False)
    axes[2].plot(history["step"], history["trade_volume"], color="#6d28d9")
    axes[2].set_title("Trade volume")
    axes[2].set_xlabel("step")
    axes[3].plot(history["step"], history["budget_surplus"], color="#374151")
    axes[3].set_title("Budget surplus")
    axes[3].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png")
    plt.close(fig)


def write_report(cfg: Config, metrics: Dict[str, float]) -> None:
    out_dir = Path(cfg.out_dir)
    lines = [
        "# Experiment Report",
        "",
        "Model: hard-constrained equilibrium-informed neural double auction.",
        "",
        "## Key Metrics",
        "",
    ]
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The construction enforces feasibility, clearing, truthful-report IR, and ex-post budget balance by design.",
            "The remaining economic tension is the welfare versus incentive-compatibility tradeoff measured by ex-post regret.",
            "For a paper-grade version, this prototype should be extended with stronger adversarial regret minimization, richer information structures, and theory for approximate IC and asymptotic efficiency.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-buyers", type=int, default=Config.n_buyers)
    parser.add_argument("--n-sellers", type=int, default=Config.n_sellers)
    parser.add_argument("--train-steps", type=int, default=Config.train_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--regret-weight", type=float, default=Config.regret_weight)
    parser.add_argument("--train-method", choices=["penalty", "dual", "aug_lagrangian"], default=Config.train_method)
    parser.add_argument("--regret-method", choices=["grid", "dense", "adv", "hybrid"], default=Config.regret_method)
    parser.add_argument("--eval-regret-method", choices=["grid", "dense", "adv", "hybrid", "exact", "audit"], default=Config.eval_regret_method)
    parser.add_argument("--regret-target", type=float, default=Config.regret_target)
    parser.add_argument("--dense-grid", type=int, default=Config.dense_grid)
    parser.add_argument("--dual-lr", type=float, default=Config.dual_lr)
    parser.add_argument("--max-regret-weight", type=float, default=Config.max_regret_weight)
    parser.add_argument("--regret-grid", type=int, default=Config.regret_grid)
    parser.add_argument("--eval-grid", type=int, default=Config.eval_grid)
    parser.add_argument("--eval-samples", type=int, default=Config.eval_samples)
    parser.add_argument("--regret-eval-samples", type=int, default=Config.regret_eval_samples)
    parser.add_argument("--adv-steps", type=int, default=Config.adv_steps)
    parser.add_argument("--adv-lr", type=float, default=Config.adv_lr)
    parser.add_argument("--adv-restarts", type=int, default=Config.adv_restarts)
    parser.add_argument("--exact-maxiter", type=int, default=Config.exact_maxiter)
    parser.add_argument("--exact-popsize", type=int, default=Config.exact_popsize)
    parser.add_argument("--exact-tol", type=float, default=Config.exact_tol)
    parser.add_argument("--augmented-rho", type=float, default=Config.augmented_rho)
    parser.add_argument("--rho-growth", type=float, default=Config.rho_growth)
    parser.add_argument("--max-augmented-rho", type=float, default=Config.max_augmented_rho)
    parser.add_argument(
        "--distribution",
        choices=["uniform", "beta_easy", "beta_hard", "asymmetric", "correlated"],
        default=Config.distribution,
    )
    parser.add_argument("--clearance-cost", type=float, default=Config.clearance_cost)
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--depth", type=int, default=Config.depth)
    parser.add_argument("--feature-mode", choices=["basic", "ranked"], default=Config.feature_mode)
    parser.add_argument("--log-every", type=int, default=Config.log_every)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--out-dir", type=str, default=Config.out_dir)
    args = parser.parse_args()
    return Config(
        n_buyers=args.n_buyers,
        n_sellers=args.n_sellers,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        regret_weight=args.regret_weight,
        train_method=args.train_method,
        regret_method=args.regret_method,
        eval_regret_method=args.eval_regret_method,
        regret_target=args.regret_target,
        dense_grid=args.dense_grid,
        dual_lr=args.dual_lr,
        max_regret_weight=args.max_regret_weight,
        regret_grid=args.regret_grid,
        eval_grid=args.eval_grid,
        eval_samples=args.eval_samples,
        regret_eval_samples=args.regret_eval_samples,
        adv_steps=args.adv_steps,
        adv_lr=args.adv_lr,
        adv_restarts=args.adv_restarts,
        exact_maxiter=args.exact_maxiter,
        exact_popsize=args.exact_popsize,
        exact_tol=args.exact_tol,
        augmented_rho=args.augmented_rho,
        rho_growth=args.rho_growth,
        max_augmented_rho=args.max_augmented_rho,
        distribution=args.distribution,
        clearance_cost=args.clearance_cost,
        hidden=args.hidden,
        depth=args.depth,
        feature_mode=args.feature_mode,
        log_every=args.log_every,
        seed=args.seed,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    _, _, metrics = train(cfg)
    write_report(cfg, metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
