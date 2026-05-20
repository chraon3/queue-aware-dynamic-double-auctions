from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from dynamic_double_auction import DynamicConfig, make_reports, mechanism_forward, public_queue_features, set_seed
from dynamic_history_audit import FEATURE_NAMES, candidate_matrix, deviation_features, load_model
from torch import nn

from econpinn_double_auction import utilities


def parse_exit_thresholds(text: str) -> torch.Tensor:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one exit threshold is required.")
    return torch.tensor(values)


def make_base_draws(cfg: DynamicConfig, episodes: int, seed: int, device: torch.device) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return {
        "tagged_buyer_value": torch.rand(episodes, generator=generator, device=device),
        "tagged_seller_cost": torch.rand(episodes, generator=generator, device=device),
        "buyer_arrival": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "seller_arrival": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "buyer_value": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "seller_cost": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "buyer_abandon": torch.rand(cfg.horizon, episodes, cfg.max_buyers, generator=generator, device=device),
        "seller_abandon": torch.rand(cfg.horizon, episodes, cfg.max_sellers, generator=generator, device=device),
    }


def expand_episode_tensor(x: torch.Tensor, strategy_count: int) -> torch.Tensor:
    return x[None, ...].expand(strategy_count, *x.shape).reshape(strategy_count * x.shape[0], *x.shape[1:])


def add_arrivals_from_draws(
    values: torch.Tensor,
    active: torch.Tensor,
    ages: torch.Tensor,
    tag: torch.Tensor,
    entry_imbalance: torch.Tensor,
    imbalance_exposure: torch.Tensor,
    queue_exposure: torch.Tensor,
    unmatched_streak: torch.Tensor,
    arrival_draw: torch.Tensor,
    type_draw: torch.Tensor,
    arrival_prob: float,
    fill_value: float,
) -> tuple[torch.Tensor, ...]:
    open_slots = active < 0.5
    has_open = open_slots.any(dim=1)
    do_arrival = (arrival_draw <= arrival_prob) & has_open
    slots = open_slots.float().argmax(dim=1).long()
    rows = torch.nonzero(do_arrival, as_tuple=False).flatten()
    if rows.numel() > 0:
        cols = slots[rows]
        values[rows, cols] = type_draw[rows]
        active[rows, cols] = 1.0
        ages[rows, cols] = 0.0
        tag[rows, cols] = 0.0
        entry_imbalance[rows, cols] = 0.0
        imbalance_exposure[rows, cols] = 0.0
        queue_exposure[rows, cols] = 0.0
        unmatched_streak[rows, cols] = 0.0
    values = torch.where(active > 0.5, values, torch.full_like(values, fill_value))
    ages = torch.where(active > 0.5, ages, torch.zeros_like(ages))
    tag = torch.where(active > 0.5, tag, torch.zeros_like(tag))
    return values, active, ages, tag, entry_imbalance, imbalance_exposure, queue_exposure, unmatched_streak


def compact_side(
    values: torch.Tensor,
    active: torch.Tensor,
    ages: torch.Tensor,
    tag: torch.Tensor,
    prev_unmatched: torch.Tensor,
    entry_imbalance: torch.Tensor,
    imbalance_exposure: torch.Tensor,
    queue_exposure: torch.Tensor,
    unmatched_streak: torch.Tensor,
    fill_value: float,
) -> tuple[torch.Tensor, ...]:
    width = active.shape[1]
    order_base = torch.arange(width, device=active.device).expand_as(active)
    sort_key = torch.where(active > 0.5, order_base, order_base + width)
    order = torch.argsort(sort_key, dim=1)
    values = values.gather(1, order)
    active = active.gather(1, order)
    ages = ages.gather(1, order)
    tag = tag.gather(1, order)
    prev_unmatched = prev_unmatched.gather(1, order)
    entry_imbalance = entry_imbalance.gather(1, order)
    imbalance_exposure = imbalance_exposure.gather(1, order)
    queue_exposure = queue_exposure.gather(1, order)
    unmatched_streak = unmatched_streak.gather(1, order)
    values = torch.where(active > 0.5, values, torch.full_like(values, fill_value))
    ages = torch.where(active > 0.5, ages, torch.zeros_like(ages))
    tag = torch.where(active > 0.5, tag, torch.zeros_like(tag))
    prev_unmatched = torch.where(active > 0.5, prev_unmatched, torch.zeros_like(prev_unmatched))
    entry_imbalance = torch.where(active > 0.5, entry_imbalance, torch.zeros_like(entry_imbalance))
    imbalance_exposure = torch.where(active > 0.5, imbalance_exposure, torch.zeros_like(imbalance_exposure))
    queue_exposure = torch.where(active > 0.5, queue_exposure, torch.zeros_like(queue_exposure))
    unmatched_streak = torch.where(active > 0.5, unmatched_streak, torch.zeros_like(unmatched_streak))
    return values, active, ages, tag, prev_unmatched, entry_imbalance, imbalance_exposure, queue_exposure, unmatched_streak


def apply_strategic_exit(
    values: torch.Tensor,
    active: torch.Tensor,
    ages: torch.Tensor,
    tag: torch.Tensor,
    prev_unmatched: torch.Tensor,
    entry_imbalance: torch.Tensor,
    imbalance_exposure: torch.Tensor,
    queue_exposure: torch.Tensor,
    unmatched_streak: torch.Tensor,
    exit_threshold_rows: torch.Tensor,
    fill_value: float,
) -> tuple[torch.Tensor, ...]:
    exit_now = (tag > 0.5) & (prev_unmatched > 0.5) & (ages >= exit_threshold_rows[:, None])
    active = torch.where(exit_now, torch.zeros_like(active), active)
    values = torch.where(exit_now, torch.full_like(values, fill_value), values)
    ages = torch.where(exit_now, torch.zeros_like(ages), ages)
    tag = torch.where(exit_now, torch.zeros_like(tag), tag)
    prev_unmatched = torch.where(exit_now, torch.zeros_like(prev_unmatched), prev_unmatched)
    entry_imbalance = torch.where(exit_now, torch.zeros_like(entry_imbalance), entry_imbalance)
    imbalance_exposure = torch.where(exit_now, torch.zeros_like(imbalance_exposure), imbalance_exposure)
    queue_exposure = torch.where(exit_now, torch.zeros_like(queue_exposure), queue_exposure)
    unmatched_streak = torch.where(exit_now, torch.zeros_like(unmatched_streak), unmatched_streak)
    return values, active, ages, tag, prev_unmatched, entry_imbalance, imbalance_exposure, queue_exposure, unmatched_streak


def tagged_period_utility(
    out: Dict[str, torch.Tensor],
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    buyer_active: torch.Tensor,
    seller_active: torch.Tensor,
    buyer_tag: torch.Tensor,
    seller_tag: torch.Tensor,
    side: str,
    cfg: DynamicConfig,
) -> torch.Tensor:
    buyer_u, seller_u = utilities(out, buyer_values, seller_costs)
    if side == "buyer":
        unmatched = buyer_active * torch.clamp(1.0 - out["buyer_alloc"], min=0.0)
        return ((buyer_u - cfg.wait_cost * unmatched) * buyer_tag).sum(dim=1)
    unmatched = seller_active * torch.clamp(1.0 - out["seller_alloc"], min=0.0)
    return ((seller_u - cfg.wait_cost * unmatched) * seller_tag).sum(dim=1)


def simulate_tagged_strategies(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    coefficients: torch.Tensor,
    exit_thresholds: torch.Tensor,
) -> torch.Tensor:
    device = torch.device(cfg.device)
    strategy_count = coefficients.shape[0]
    episodes = draws["tagged_buyer_value"].shape[0]
    row_count = strategy_count * episodes
    coeff_rows = coefficients[:, None, :].expand(strategy_count, episodes, len(FEATURE_NAMES)).reshape(row_count, len(FEATURE_NAMES))
    exit_rows = exit_thresholds[:, None].expand(strategy_count, episodes).reshape(row_count).to(device)

    buyer_values = torch.zeros(row_count, cfg.max_buyers, device=device)
    seller_costs = torch.ones(row_count, cfg.max_sellers, device=device)
    buyer_active = torch.zeros_like(buyer_values)
    seller_active = torch.zeros_like(seller_costs)
    buyer_ages = torch.zeros_like(buyer_values)
    seller_ages = torch.zeros_like(seller_costs)
    buyer_tag = torch.zeros_like(buyer_values)
    seller_tag = torch.zeros_like(seller_costs)
    buyer_prev_unmatched = torch.zeros_like(buyer_values)
    seller_prev_unmatched = torch.zeros_like(seller_costs)
    buyer_entry_imbalance = torch.zeros_like(buyer_values)
    seller_entry_imbalance = torch.zeros_like(seller_costs)
    buyer_imbalance_exposure = torch.zeros_like(buyer_values)
    seller_imbalance_exposure = torch.zeros_like(seller_costs)
    buyer_queue_exposure = torch.zeros_like(buyer_values)
    seller_queue_exposure = torch.zeros_like(seller_costs)
    buyer_unmatched_streak = torch.zeros_like(buyer_values)
    seller_unmatched_streak = torch.zeros_like(seller_costs)

    if side == "buyer":
        buyer_values[:, 0] = expand_episode_tensor(draws["tagged_buyer_value"], strategy_count)
        buyer_active[:, 0] = 1.0
        buyer_tag[:, 0] = 1.0
    else:
        seller_costs[:, 0] = expand_episode_tensor(draws["tagged_seller_cost"], strategy_count)
        seller_active[:, 0] = 1.0
        seller_tag[:, 0] = 1.0

    utilities_by_row = torch.zeros(row_count, device=device)
    for t in range(cfg.horizon):
        buyer_values, buyer_active, buyer_ages, buyer_tag, buyer_prev_unmatched, buyer_entry_imbalance, buyer_imbalance_exposure, buyer_queue_exposure, buyer_unmatched_streak = compact_side(
            buyer_values,
            buyer_active,
            buyer_ages,
            buyer_tag,
            buyer_prev_unmatched,
            buyer_entry_imbalance,
            buyer_imbalance_exposure,
            buyer_queue_exposure,
            buyer_unmatched_streak,
            0.0,
        )
        seller_costs, seller_active, seller_ages, seller_tag, seller_prev_unmatched, seller_entry_imbalance, seller_imbalance_exposure, seller_queue_exposure, seller_unmatched_streak = compact_side(
            seller_costs,
            seller_active,
            seller_ages,
            seller_tag,
            seller_prev_unmatched,
            seller_entry_imbalance,
            seller_imbalance_exposure,
            seller_queue_exposure,
            seller_unmatched_streak,
            1.0,
        )

        if side == "buyer":
            buyer_values, buyer_active, buyer_ages, buyer_tag, buyer_prev_unmatched, buyer_entry_imbalance, buyer_imbalance_exposure, buyer_queue_exposure, buyer_unmatched_streak = apply_strategic_exit(
                buyer_values,
                buyer_active,
                buyer_ages,
                buyer_tag,
                buyer_prev_unmatched,
                buyer_entry_imbalance,
                buyer_imbalance_exposure,
                buyer_queue_exposure,
                buyer_unmatched_streak,
                exit_rows,
                0.0,
            )
        else:
            seller_costs, seller_active, seller_ages, seller_tag, seller_prev_unmatched, seller_entry_imbalance, seller_imbalance_exposure, seller_queue_exposure, seller_unmatched_streak = apply_strategic_exit(
                seller_costs,
                seller_active,
                seller_ages,
                seller_tag,
                seller_prev_unmatched,
                seller_entry_imbalance,
                seller_imbalance_exposure,
                seller_queue_exposure,
                seller_unmatched_streak,
                exit_rows,
                1.0,
            )

        pre_buyer_active = buyer_active.clone()
        pre_seller_active = seller_active.clone()
        buyer_values, buyer_active, buyer_ages, buyer_tag, buyer_entry_imbalance, buyer_imbalance_exposure, buyer_queue_exposure, buyer_unmatched_streak = add_arrivals_from_draws(
            buyer_values,
            buyer_active,
            buyer_ages,
            buyer_tag,
            buyer_entry_imbalance,
            buyer_imbalance_exposure,
            buyer_queue_exposure,
            buyer_unmatched_streak,
            expand_episode_tensor(draws["buyer_arrival"][t], strategy_count),
            expand_episode_tensor(draws["buyer_value"][t], strategy_count),
            cfg.arrival_prob_buyer,
            0.0,
        )
        seller_costs, seller_active, seller_ages, seller_tag, seller_entry_imbalance, seller_imbalance_exposure, seller_queue_exposure, seller_unmatched_streak = add_arrivals_from_draws(
            seller_costs,
            seller_active,
            seller_ages,
            seller_tag,
            seller_entry_imbalance,
            seller_imbalance_exposure,
            seller_queue_exposure,
            seller_unmatched_streak,
            expand_episode_tensor(draws["seller_arrival"][t], strategy_count),
            expand_episode_tensor(draws["seller_cost"][t], strategy_count),
            cfg.arrival_prob_seller,
            1.0,
        )

        new_buyers = (buyer_active > 0.5) & (pre_buyer_active < 0.5)
        new_sellers = (seller_active > 0.5) & (pre_seller_active < 0.5)
        buyer_count = buyer_active.sum(dim=1)
        seller_count = seller_active.sum(dim=1)
        total_capacity = max(cfg.max_buyers + cfg.max_sellers, 1)
        current_imbalance = (buyer_count - seller_count) / total_capacity
        current_queue = (buyer_count + seller_count) / total_capacity
        buyer_side_imbalance = current_imbalance
        seller_side_imbalance = -current_imbalance
        buyer_entry_imbalance = torch.where(new_buyers, buyer_side_imbalance[:, None], buyer_entry_imbalance)
        seller_entry_imbalance = torch.where(new_sellers, seller_side_imbalance[:, None], seller_entry_imbalance)

        buyer_reports, seller_reports = make_reports(buyer_values, seller_costs, buyer_active, seller_active)
        if side == "buyer":
            features = deviation_features(
                buyer_active,
                buyer_ages,
                buyer_prev_unmatched,
                buyer_entry_imbalance,
                buyer_imbalance_exposure,
                buyer_queue_exposure,
                buyer_unmatched_streak,
                buyer_side_imbalance,
                cfg,
            )
            shift = (features * coeff_rows[:, None, :]).sum(dim=2)
            tagged_reports = torch.clamp(buyer_values + shift, 0.0, 1.0)
            buyer_reports = torch.where(buyer_tag > 0.5, tagged_reports, buyer_reports)
        else:
            features = deviation_features(
                seller_active,
                seller_ages,
                seller_prev_unmatched,
                seller_entry_imbalance,
                seller_imbalance_exposure,
                seller_queue_exposure,
                seller_unmatched_streak,
                seller_side_imbalance,
                cfg,
            )
            shift = (features * coeff_rows[:, None, :]).sum(dim=2)
            tagged_reports = torch.clamp(seller_costs + shift, 0.0, 1.0)
            seller_reports = torch.where(seller_tag > 0.5, tagged_reports, seller_reports)

        public_state = public_queue_features(buyer_active, seller_active, buyer_ages, seller_ages, t, cfg)
        out = mechanism_forward(model, buyer_reports, seller_reports, public_state)
        active_pair = buyer_active[:, :, None] * seller_active[:, None, :]
        match = out["match"] * active_pair
        out = dict(out)
        out["match"] = match
        out["buyer_alloc"] = torch.clamp(match.sum(dim=2), 0.0, 1.0)
        out["seller_alloc"] = torch.clamp(match.sum(dim=1), 0.0, 1.0)
        out["buyer_payments"] = out["buyer_payments"] * buyer_active
        out["seller_transfers"] = out["seller_transfers"] * seller_active
        utilities_by_row += (cfg.discount**t) * tagged_period_utility(
            out,
            buyer_values,
            seller_costs,
            buyer_active,
            seller_active,
            buyer_tag,
            seller_tag,
            side,
            cfg,
        )

        buyer_alloc = out["buyer_alloc"]
        seller_alloc = out["seller_alloc"]
        unmatched_buyers = buyer_active * torch.clamp(1.0 - buyer_alloc, min=0.0)
        unmatched_sellers = seller_active * torch.clamp(1.0 - seller_alloc, min=0.0)
        depart_buyers = buyer_alloc.detach() > 0.5
        depart_sellers = seller_alloc.detach() > 0.5
        buyer_ages = buyer_ages + unmatched_buyers.detach()
        seller_ages = seller_ages + unmatched_sellers.detach()
        buyer_abandon_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * buyer_ages / max(cfg.max_patience, 1), 0.0, 1.0)
        seller_abandon_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * seller_ages / max(cfg.max_patience, 1), 0.0, 1.0)
        buyer_abandon_draw = expand_episode_tensor(draws["buyer_abandon"][t], strategy_count)
        seller_abandon_draw = expand_episode_tensor(draws["seller_abandon"][t], strategy_count)
        buyer_abandon = ((buyer_abandon_draw < buyer_abandon_prob) | (buyer_ages >= cfg.max_patience)) & (buyer_active > 0.5) & (~depart_buyers)
        seller_abandon = ((seller_abandon_draw < seller_abandon_prob) | (seller_ages >= cfg.max_patience)) & (seller_active > 0.5) & (~depart_sellers)
        surviving_buyers = (buyer_active > 0.5) & (~depart_buyers) & (~buyer_abandon)
        surviving_sellers = (seller_active > 0.5) & (~depart_sellers) & (~seller_abandon)
        buyer_imbalance_exposure = torch.where(surviving_buyers, buyer_imbalance_exposure + buyer_side_imbalance[:, None], torch.zeros_like(buyer_imbalance_exposure))
        seller_imbalance_exposure = torch.where(surviving_sellers, seller_imbalance_exposure + seller_side_imbalance[:, None], torch.zeros_like(seller_imbalance_exposure))
        buyer_queue_exposure = torch.where(surviving_buyers, buyer_queue_exposure + current_queue[:, None], torch.zeros_like(buyer_queue_exposure))
        seller_queue_exposure = torch.where(surviving_sellers, seller_queue_exposure + current_queue[:, None], torch.zeros_like(seller_queue_exposure))
        buyer_unmatched_streak = torch.where(surviving_buyers, buyer_unmatched_streak + (unmatched_buyers.detach() > 0.5).float(), torch.zeros_like(buyer_unmatched_streak))
        seller_unmatched_streak = torch.where(surviving_sellers, seller_unmatched_streak + (unmatched_sellers.detach() > 0.5).float(), torch.zeros_like(seller_unmatched_streak))
        buyer_active = torch.where(depart_buyers | buyer_abandon, torch.zeros_like(buyer_active), buyer_active)
        seller_active = torch.where(depart_sellers | seller_abandon, torch.zeros_like(seller_active), seller_active)
        buyer_prev_unmatched = torch.where(buyer_active > 0.5, unmatched_buyers.detach(), torch.zeros_like(buyer_prev_unmatched))
        seller_prev_unmatched = torch.where(seller_active > 0.5, unmatched_sellers.detach(), torch.zeros_like(seller_prev_unmatched))
        buyer_values = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
        seller_costs = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
        buyer_ages = torch.where(buyer_active > 0.5, buyer_ages, torch.zeros_like(buyer_ages))
        seller_ages = torch.where(seller_active > 0.5, seller_ages, torch.zeros_like(seller_ages))
        buyer_tag = torch.where(buyer_active > 0.5, buyer_tag, torch.zeros_like(buyer_tag))
        seller_tag = torch.where(seller_active > 0.5, seller_tag, torch.zeros_like(seller_tag))

    return utilities_by_row.reshape(strategy_count, episodes)


def continuation_regret_components_for_side(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    coefficients: torch.Tensor,
    exit_thresholds: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(cfg.device)
    no_deviation = torch.zeros(1, len(FEATURE_NAMES), device=device)
    no_exit = torch.tensor([max(cfg.horizon + cfg.max_patience + 10, 99)], device=device, dtype=torch.float32)
    truthful = simulate_tagged_strategies(model, cfg, draws, side, no_deviation, no_exit)[0]
    chunk_utilities = []
    for start in range(0, coefficients.shape[0], chunk_size):
        end = min(start + chunk_size, coefficients.shape[0])
        chunk_utilities.append(simulate_tagged_strategies(model, cfg, draws, side, coefficients[start:end], exit_thresholds[start:end]))
    deviating = torch.cat(chunk_utilities, dim=0)
    best_values, best_indices = deviating.max(dim=0)
    all_regret = torch.relu(best_values - truthful)
    no_exit_mask = exit_thresholds >= max(cfg.horizon + cfg.max_patience + 10, 99)
    if torch.any(no_exit_mask):
        no_exit_regret = torch.relu(deviating[no_exit_mask].max(dim=0).values - truthful)
    else:
        no_exit_regret = all_regret
    return all_regret, no_exit_regret, truthful, best_values, best_indices


def continuation_regret_for_side(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    coefficients: torch.Tensor,
    exit_thresholds: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_regret, no_exit_regret, _, _, _ = continuation_regret_components_for_side(
        model,
        cfg,
        draws,
        side,
        coefficients,
        exit_thresholds,
        chunk_size,
    )
    return all_regret, no_exit_regret


def continuation_regret_detail_rows_for_side(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    coefficients: torch.Tensor,
    exit_thresholds: torch.Tensor,
    chunk_size: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[Dict[str, float | str]]]:
    all_regret, no_exit_regret, truthful, best_values, best_indices = continuation_regret_components_for_side(
        model,
        cfg,
        draws,
        side,
        coefficients,
        exit_thresholds,
        chunk_size,
    )
    rows: list[Dict[str, float | str]] = []
    if top_k <= 0 or all_regret.numel() == 0:
        return all_regret, no_exit_regret, rows

    detail_count = min(top_k, all_regret.numel())
    exit_threshold_count = int(torch.unique(exit_thresholds).numel())
    no_exit_level = max(cfg.horizon + cfg.max_patience + 10, 99)
    top_values, top_episodes = torch.topk(all_regret.detach(), detail_count)
    for rank, (regret_value, episode) in enumerate(zip(top_values, top_episodes), start=1):
        episode_idx = int(episode.item())
        strategy_index = int(best_indices[episode_idx].detach().item())
        report_index = strategy_index // max(exit_threshold_count, 1)
        exit_index = strategy_index % max(exit_threshold_count, 1)
        threshold = float(exit_thresholds[strategy_index].detach().item())
        coefficients_row = coefficients[strategy_index].detach().cpu()
        row: Dict[str, float | str] = {
            "side": side,
            "rank_within_side": float(rank),
            "episode": float(episode_idx),
            "regret": float(regret_value.item()),
            "truthful_utility": float(truthful[episode_idx].detach().item()),
            "best_utility": float(best_values[episode_idx].detach().item()),
            "strategy_index": float(strategy_index),
            "report_index": float(report_index),
            "exit_index": float(exit_index),
            "exit_threshold": threshold,
            "is_no_exit": float(threshold >= no_exit_level),
        }
        for feature_name, coefficient in zip(FEATURE_NAMES, coefficients_row.tolist()):
            row[f"coef_{feature_name}"] = float(coefficient)
        rows.append(row)
    return all_regret, no_exit_regret, rows


def build_strategy_set(
    family: str,
    radius: float,
    grid: int,
    history_draws: int,
    exit_thresholds: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    reports = candidate_matrix(family, radius, grid, history_draws, seed, device)
    thresholds = exit_thresholds.to(device=device, dtype=torch.float32)
    coefficients = reports.repeat_interleave(thresholds.numel(), dim=0)
    expanded_thresholds = thresholds.repeat(reports.shape[0])
    return coefficients, expanded_thresholds, reports.shape[0]


def audit_run(
    run_dir: Path,
    episodes: int,
    grid: int,
    radius: float,
    family: str,
    history_draws: int,
    exit_thresholds: torch.Tensor,
    chunk_size: int,
    seed_offset: int,
    detail_top_k: int = 0,
    sample_path: Path | None = None,
) -> tuple[Dict[str, float | str], list[Dict[str, float | str]]]:
    cfg, model = load_model(run_dir)
    cfg = replace(cfg, eval_episodes=episodes)
    set_seed(cfg.seed + seed_offset)
    device = torch.device(cfg.device)
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
        if detail_top_k > 0:
            buyer_regret, buyer_no_exit_regret, buyer_details = continuation_regret_detail_rows_for_side(
                model,
                cfg,
                buyer_draws,
                "buyer",
                coefficients,
                strategy_exit_thresholds,
                chunk_size,
                detail_top_k,
            )
            seller_regret, seller_no_exit_regret, seller_details = continuation_regret_detail_rows_for_side(
                model,
                cfg,
                seller_draws,
                "seller",
                coefficients,
                strategy_exit_thresholds,
                chunk_size,
                detail_top_k,
            )
        else:
            buyer_regret, buyer_no_exit_regret = continuation_regret_for_side(model, cfg, buyer_draws, "buyer", coefficients, strategy_exit_thresholds, chunk_size)
            seller_regret, seller_no_exit_regret = continuation_regret_for_side(model, cfg, seller_draws, "seller", coefficients, strategy_exit_thresholds, chunk_size)
            buyer_details = []
            seller_details = []
    all_regret = torch.cat([buyer_regret, seller_regret])
    no_exit_regret = torch.cat([buyer_no_exit_regret, seller_no_exit_regret])
    if sample_path is not None:
        sample_rows = []
        for side, regret, no_exit_side_regret in [
            ("buyer", buyer_regret, buyer_no_exit_regret),
            ("seller", seller_regret, seller_no_exit_regret),
        ]:
            regret_values = regret.detach().cpu().tolist()
            no_exit_values = no_exit_side_regret.detach().cpu().tolist()
            for episode_idx, (regret_value, no_exit_value) in enumerate(zip(regret_values, no_exit_values)):
                sample_rows.append(
                    {
                        "run": run_dir.name,
                        "audit_family": family,
                        "side": side,
                        "episode": episode_idx,
                        "seed_offset": float(seed_offset),
                        "grid": float(grid),
                        "radius": float(radius),
                        "history_draws": float(history_draws if family.startswith("history") else 0),
                        "exit_threshold_count": float(exit_thresholds.numel()),
                        "continuation_strategy_count": float(coefficients.shape[0]),
                        "continuation_regret": float(regret_value),
                        "no_exit_continuation_regret": float(no_exit_value),
                    }
                )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    row: Dict[str, float | str] = {
        "run": run_dir.name,
        "audit_family": family,
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
    detail_rows = []
    for detail in buyer_details + seller_details:
        detail.update(
            {
                "run": run_dir.name,
                "audit_family": family,
                "grid": float(grid),
                "radius": float(radius),
                "history_draws": float(history_draws if family.startswith("history") else 0),
                "seed_offset": float(seed_offset),
            }
        )
        detail_rows.append(detail)
    return row, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--family", choices=["age", "state", "history", "history_nonlinear", "history_piecewise"], default="history")
    parser.add_argument("--history-draws", type=int, default=48)
    parser.add_argument("--exit-thresholds", default="2,3,4,99")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--out-dir", default="experiments/dynamic_continuation_audit")
    parser.add_argument("--save-details", action="store_true", help="Write top continuation-regret episodes and their argmax deviation strategies.")
    parser.add_argument("--save-samples", action="store_true", help="Write episode-level continuation-regret samples for Monte Carlo uncertainty diagnostics.")
    parser.add_argument("--top-k-details", type=int, default=5, help="Per-side top-k episodes to save when --save-details is enabled.")
    parser.add_argument("--seed-offset", type=int, default=80_000, help="Base random seed offset for continuation-audit draw generation.")
    parser.add_argument("--seed-stride", type=int, default=997, help="Stride added to the seed offset across runs unless --paired-draws is used.")
    parser.add_argument("--paired-draws", action="store_true", help="Use common random continuation-audit draws for every run in the comparison.")
    args = parser.parse_args()

    exit_thresholds = parse_exit_thresholds(args.exit_thresholds)
    run_dirs = []
    for pattern in args.runs:
        run_dirs.extend(sorted(Path().glob(pattern)))
    run_dirs = [path for path in run_dirs if (path / "model.pt").exists()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    detail_rows = []
    sample_paths = []
    detail_top_k = max(args.top_k_details, 0) if args.save_details else 0
    for idx, run_dir in enumerate(run_dirs):
        print(f"Continuation auditing {run_dir.name}", flush=True)
        seed_offset = args.seed_offset if args.paired_draws else args.seed_offset + idx * args.seed_stride
        sample_path = out_dir / f"{run_dir.name}_continuation_samples.csv" if args.save_samples else None
        row, run_detail_rows = audit_run(
            run_dir,
            args.episodes,
            args.grid,
            args.radius,
            args.family,
            args.history_draws,
            exit_thresholds,
            args.chunk_size,
            seed_offset,
            detail_top_k,
            sample_path,
        )
        rows.append(row)
        detail_rows.extend(run_detail_rows)
        if sample_path is not None:
            sample_paths.append(sample_path)
        (out_dir / f"{run_dir.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "continuation_audit_by_run.csv", index=False)
    if args.save_details:
        details = pd.DataFrame(detail_rows)
        if not details.empty:
            details = details.sort_values(["run", "side", "rank_within_side"])
        details.to_csv(out_dir / "continuation_audit_worst_cases.csv", index=False)
    if args.save_samples:
        sample_frames = [pd.read_csv(path) for path in sample_paths if path.exists()]
        samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
        samples.to_csv(out_dir / "continuation_audit_samples.csv", index=False)
    numeric = raw.drop(columns=["run", "audit_family"])
    summary = numeric.agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    summary.to_csv(out_dir / "continuation_audit_summary.csv", index=False)
    print(raw.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
