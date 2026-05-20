from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch import nn

from dynamic_continuation_audit import (
    add_arrivals_from_draws,
    compact_side,
    expand_episode_tensor,
    make_base_draws,
    parse_exit_thresholds,
    simulate_tagged_strategies,
    tagged_period_utility,
)
from dynamic_double_auction import DynamicConfig, make_reports, mechanism_forward, public_queue_features, set_seed
from dynamic_history_audit import FEATURE_NAMES, deviation_features, load_model


EXIT_FEATURE_DIM = len(FEATURE_NAMES) + 2


def raw_to_policy(raw: torch.Tensor, report_radius: float, exit_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    report_dim = len(FEATURE_NAMES)
    report = report_radius * torch.tanh(raw[:, :report_dim])
    exit_coefficients = exit_scale * torch.tanh(raw[:, report_dim:])
    return report, exit_coefficients


def exit_inputs(
    history_features: torch.Tensor,
    private_type: torch.Tensor,
    active: torch.Tensor,
    t: int,
    cfg: DynamicConfig,
) -> torch.Tensor:
    time = torch.full_like(active, float(t) / max(cfg.horizon - 1, 1))
    return torch.cat([history_features, private_type[:, :, None], time[:, :, None]], dim=2)


def truthful_utility(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
) -> torch.Tensor:
    device = torch.device(cfg.device)
    no_deviation = torch.zeros(1, len(FEATURE_NAMES), device=device)
    no_exit = torch.tensor([max(cfg.horizon + cfg.max_patience + 10, 99)], device=device, dtype=torch.float32)
    with torch.no_grad():
        return simulate_tagged_strategies(model, cfg, draws, side, no_deviation, no_exit)[0]


def apply_learned_exit(
    values: torch.Tensor,
    active: torch.Tensor,
    ages: torch.Tensor,
    tag: torch.Tensor,
    prev_unmatched: torch.Tensor,
    entry_imbalance: torch.Tensor,
    imbalance_exposure: torch.Tensor,
    queue_exposure: torch.Tensor,
    unmatched_streak: torch.Tensor,
    exit_now: torch.Tensor,
    fill_value: float,
) -> tuple[torch.Tensor, ...]:
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


def simulate_report_exit_policies(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    report_coefficients: torch.Tensor,
    exit_coefficients: torch.Tensor,
) -> torch.Tensor:
    device = torch.device(cfg.device)
    strategy_count = report_coefficients.shape[0]
    episodes = draws["tagged_buyer_value"].shape[0]
    row_count = strategy_count * episodes
    report_rows = report_coefficients[:, None, :].expand(strategy_count, episodes, len(FEATURE_NAMES)).reshape(row_count, len(FEATURE_NAMES))
    exit_rows = exit_coefficients[:, None, :].expand(strategy_count, episodes, EXIT_FEATURE_DIM).reshape(row_count, EXIT_FEATURE_DIM)

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

        buyer_count_pre = buyer_active.sum(dim=1)
        seller_count_pre = seller_active.sum(dim=1)
        total_capacity = max(cfg.max_buyers + cfg.max_sellers, 1)
        imbalance_pre = (buyer_count_pre - seller_count_pre) / total_capacity
        if side == "buyer":
            history_features = deviation_features(
                buyer_active,
                buyer_ages,
                buyer_prev_unmatched,
                buyer_entry_imbalance,
                buyer_imbalance_exposure,
                buyer_queue_exposure,
                buyer_unmatched_streak,
                imbalance_pre,
                cfg,
            )
            inputs = exit_inputs(history_features, buyer_values, buyer_active, t, cfg)
            score = torch.einsum("bd,bnd->bn", exit_rows, inputs)
            exit_now = (buyer_tag > 0.5) & (buyer_prev_unmatched > 0.5) & (score > 0.0)
            buyer_values, buyer_active, buyer_ages, buyer_tag, buyer_prev_unmatched, buyer_entry_imbalance, buyer_imbalance_exposure, buyer_queue_exposure, buyer_unmatched_streak = apply_learned_exit(
                buyer_values,
                buyer_active,
                buyer_ages,
                buyer_tag,
                buyer_prev_unmatched,
                buyer_entry_imbalance,
                buyer_imbalance_exposure,
                buyer_queue_exposure,
                buyer_unmatched_streak,
                exit_now,
                0.0,
            )
        else:
            history_features = deviation_features(
                seller_active,
                seller_ages,
                seller_prev_unmatched,
                seller_entry_imbalance,
                seller_imbalance_exposure,
                seller_queue_exposure,
                seller_unmatched_streak,
                -imbalance_pre,
                cfg,
            )
            inputs = exit_inputs(history_features, seller_costs, seller_active, t, cfg)
            score = torch.einsum("bd,bnd->bn", exit_rows, inputs)
            exit_now = (seller_tag > 0.5) & (seller_prev_unmatched > 0.5) & (score > 0.0)
            seller_costs, seller_active, seller_ages, seller_tag, seller_prev_unmatched, seller_entry_imbalance, seller_imbalance_exposure, seller_queue_exposure, seller_unmatched_streak = apply_learned_exit(
                seller_costs,
                seller_active,
                seller_ages,
                seller_tag,
                seller_prev_unmatched,
                seller_entry_imbalance,
                seller_imbalance_exposure,
                seller_queue_exposure,
                seller_unmatched_streak,
                exit_now,
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
            shift = (features * report_rows[:, None, :]).sum(dim=2)
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
            shift = (features * report_rows[:, None, :]).sum(dim=2)
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


def objective_from_gain(gain: torch.Tensor, tail_weight: float, tail_alpha: float) -> torch.Tensor:
    objective = gain.mean(dim=1)
    if tail_weight > 0.0 and gain.shape[1] > 1:
        top_count = max(1, math.ceil((1.0 - tail_alpha) * gain.shape[1]))
        objective = objective + tail_weight * torch.topk(gain, top_count, dim=1).values.mean(dim=1)
    return objective


def cem_search(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    truthful: torch.Tensor,
    population: int,
    iterations: int,
    elite_count: int,
    keep_policies: int,
    report_radius: float,
    exit_scale: float,
    initial_std: float,
    min_std: float,
    tail_weight: float,
    tail_alpha: float,
    seed: int,
) -> torch.Tensor:
    device = torch.device(cfg.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    dim = len(FEATURE_NAMES) + EXIT_FEATURE_DIM
    mean = torch.zeros(dim, device=device)
    std = torch.full((dim,), initial_std, device=device)
    best_raw: list[torch.Tensor] = []
    best_scores: list[torch.Tensor] = []
    for _iteration in range(iterations):
        noise = torch.randn(population, dim, generator=generator, device=device)
        raw = mean[None, :] + std[None, :] * noise
        raw[0] = mean
        report, exit_coefficients = raw_to_policy(raw, report_radius, exit_scale)
        with torch.no_grad():
            utility = simulate_report_exit_policies(model, cfg, draws, side, report, exit_coefficients)
            gain = utility - truthful[None, :]
            scores = objective_from_gain(gain, tail_weight, tail_alpha)
        elite_n = min(elite_count, population)
        elite_indices = torch.topk(scores, elite_n).indices
        elites = raw[elite_indices]
        mean = elites.mean(dim=0)
        std = torch.clamp(elites.std(dim=0, unbiased=False), min=min_std)
        best_raw.append(raw.detach())
        best_scores.append(scores.detach())
    all_raw = torch.cat(best_raw, dim=0)
    all_scores = torch.cat(best_scores, dim=0)
    keep = min(keep_policies, all_raw.shape[0])
    keep_indices = torch.topk(all_scores, keep).indices
    return all_raw[keep_indices].detach()


def regret_summary(regret: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_regret": float(regret.mean().item()),
        f"{prefix}_p95_regret": float(torch.quantile(regret, 0.95).item()),
        f"{prefix}_max_regret": float(regret.max().item()),
        f"{prefix}_regret_count": float(regret.numel()),
    }


def evaluate_raw_policies(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    truthful: torch.Tensor,
    raw: torch.Tensor,
    report_radius: float,
    exit_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    regrets = []
    for start in range(0, raw.shape[0], chunk_size):
        chunk = raw[start : start + chunk_size]
        report, exit_coefficients = raw_to_policy(chunk, report_radius, exit_scale)
        with torch.no_grad():
            utility = simulate_report_exit_policies(model, cfg, draws, side, report, exit_coefficients)
        regrets.append(utility - truthful[None, :])
    gain = torch.cat(regrets, dim=0)
    return torch.relu(gain.max(dim=0).values)


def audit_side(
    model: nn.Module,
    cfg: DynamicConfig,
    side: str,
    train_draws: Dict[str, torch.Tensor],
    val_draws: Dict[str, torch.Tensor],
    test_draws: Dict[str, torch.Tensor],
    population: int,
    iterations: int,
    elite_count: int,
    keep_policies: int,
    restarts: int,
    report_radius: float,
    exit_scale: float,
    initial_std: float,
    min_std: float,
    tail_weight: float,
    tail_alpha: float,
    chunk_size: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    train_truthful = truthful_utility(model, cfg, train_draws, side)
    val_truthful = truthful_utility(model, cfg, val_draws, side)
    test_truthful = truthful_utility(model, cfg, test_draws, side)
    raw_policies = []
    for restart in range(restarts):
        raw_policies.append(
            cem_search(
                model,
                cfg,
                train_draws,
                side,
                train_truthful,
                population,
                iterations,
                elite_count,
                keep_policies,
                report_radius,
                exit_scale,
                initial_std,
                min_std,
                tail_weight,
                tail_alpha,
                seed + 997 * restart,
            )
        )
    raw = torch.cat(raw_policies, dim=0)
    train_regret = evaluate_raw_policies(model, cfg, train_draws, side, train_truthful, raw, report_radius, exit_scale, chunk_size)
    val_regret = evaluate_raw_policies(model, cfg, val_draws, side, val_truthful, raw, report_radius, exit_scale, chunk_size)
    test_regret = evaluate_raw_policies(model, cfg, test_draws, side, test_truthful, raw, report_radius, exit_scale, chunk_size)
    row: dict[str, float] = {}
    row.update(regret_summary(train_regret, f"{side}_train"))
    row.update(regret_summary(val_regret, f"{side}_val"))
    row.update(regret_summary(test_regret, f"{side}_test"))

    sample_rows: list[dict[str, float | str]] = []
    for split, regret in [("train", train_regret), ("val", val_regret), ("test", test_regret)]:
        for episode, value in enumerate(regret.detach().cpu().tolist()):
            sample_rows.append(
                {
                    "side": side,
                    "split": split,
                    "episode": float(episode),
                    "learned_report_exit_regret": float(value),
                }
            )
    return row, sample_rows


def audit_run(
    run_dir: Path,
    train_episodes: int,
    val_episodes: int,
    test_episodes: int,
    population: int,
    iterations: int,
    elite_count: int,
    keep_policies: int,
    restarts: int,
    report_radius: float,
    exit_scale: float,
    initial_std: float,
    min_std: float,
    tail_weight: float,
    tail_alpha: float,
    chunk_size: int,
    seed_offset: int,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    cfg, model = load_model(run_dir)
    cfg = replace(cfg, eval_episodes=max(train_episodes, val_episodes, test_episodes))
    device = torch.device(cfg.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    set_seed(cfg.seed + seed_offset)
    train_draws = make_base_draws(cfg, train_episodes, cfg.seed + seed_offset + 11, device)
    val_draws = make_base_draws(cfg, val_episodes, cfg.seed + seed_offset + 17, device)
    test_draws = make_base_draws(cfg, test_episodes, cfg.seed + seed_offset + 29, device)

    buyer_row, buyer_samples = audit_side(
        model,
        cfg,
        "buyer",
        train_draws,
        val_draws,
        test_draws,
        population,
        iterations,
        elite_count,
        keep_policies,
        restarts,
        report_radius,
        exit_scale,
        initial_std,
        min_std,
        tail_weight,
        tail_alpha,
        chunk_size,
        cfg.seed + seed_offset + 101,
    )
    seller_row, seller_samples = audit_side(
        model,
        cfg,
        "seller",
        train_draws,
        val_draws,
        test_draws,
        population,
        iterations,
        elite_count,
        keep_policies,
        restarts,
        report_radius,
        exit_scale,
        initial_std,
        min_std,
        tail_weight,
        tail_alpha,
        chunk_size,
        cfg.seed + seed_offset + 503,
    )

    sample_rows = []
    for sample in buyer_samples + seller_samples:
        sample.update({"run": run_dir.name})
        sample_rows.append(sample)

    combined: dict[str, float] = {}
    samples = pd.DataFrame(sample_rows)
    for split in ["train", "val", "test"]:
        values = torch.tensor(
            samples.loc[samples["split"] == split, "learned_report_exit_regret"].to_numpy(),
            device=device,
            dtype=torch.float32,
        )
        combined.update(regret_summary(values, f"learned_report_exit_{split}"))

    row: dict[str, float | str] = {
        "run": run_dir.name,
        "train_episodes_per_side": float(train_episodes),
        "val_episodes_per_side": float(val_episodes),
        "test_episodes_per_side": float(test_episodes),
        "population": float(population),
        "iterations": float(iterations),
        "elite_count": float(elite_count),
        "keep_policies": float(keep_policies),
        "restarts": float(restarts),
        "learned_strategy_count_per_side": float(keep_policies * restarts),
        "report_radius": float(report_radius),
        "exit_scale": float(exit_scale),
        "initial_std": float(initial_std),
        "min_std": float(min_std),
        "tail_weight": float(tail_weight),
        "tail_alpha": float(tail_alpha),
        "seed_offset": float(seed_offset),
    }
    row.update(buyer_row)
    row.update(seller_row)
    row.update(combined)
    return row, sample_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--train-episodes", type=int, default=48)
    parser.add_argument("--val-episodes", type=int, default=40)
    parser.add_argument("--test-episodes", type=int, default=80)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--keep-policies", type=int, default=6)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--report-radius", type=float, default=0.35)
    parser.add_argument("--exit-scale", type=float, default=4.0)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=0.15)
    parser.add_argument("--tail-weight", type=float, default=0.7)
    parser.add_argument("--tail-alpha", type=float, default=0.8)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--seed-offset", type=int, default=360_000)
    parser.add_argument("--seed-stride", type=int, default=997)
    parser.add_argument("--out-dir", default="experiments/learned_report_exit_auditor")
    args = parser.parse_args()

    run_dirs = []
    for pattern in args.runs:
        run_dirs.extend(sorted(Path().glob(pattern)))
    run_dirs = [path for path in run_dirs if (path / "model.pt").exists()]
    if not run_dirs:
        raise FileNotFoundError("No run directories with model.pt were found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    samples = []
    for idx, run_dir in enumerate(run_dirs):
        print(f"Searching learned report-exit auditor for {run_dir.name}", flush=True)
        row, sample_rows = audit_run(
            run_dir,
            args.train_episodes,
            args.val_episodes,
            args.test_episodes,
            args.population,
            args.iterations,
            args.elite_count,
            args.keep_policies,
            args.restarts,
            args.report_radius,
            args.exit_scale,
            args.initial_std,
            args.min_std,
            args.tail_weight,
            args.tail_alpha,
            args.chunk_size,
            args.seed_offset + idx * args.seed_stride,
        )
        rows.append(row)
        samples.extend(sample_rows)
        (out_dir / f"{run_dir.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "learned_report_exit_by_run.csv", index=False)
    pd.DataFrame(samples).to_csv(out_dir / "learned_report_exit_samples.csv", index=False)
    numeric = raw.drop(columns=["run"])
    summary = numeric.agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    summary.to_csv(out_dir / "learned_report_exit_summary.csv", index=False)
    print(raw.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
