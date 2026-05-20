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
    apply_strategic_exit,
    compact_side,
    make_base_draws,
    parse_exit_thresholds,
    simulate_tagged_strategies,
    tagged_period_utility,
)
from dynamic_double_auction import DynamicConfig, make_reports, mechanism_forward, public_queue_features, set_seed
from dynamic_history_audit import FEATURE_NAMES, deviation_features, load_model


class NeuralReportAuditor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, radius: float) -> None:
        super().__init__()
        self.radius = float(radius)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.radius * torch.tanh(self.net(features)).squeeze(-1)


def expand_time_feature(active: torch.Tensor, t: int, cfg: DynamicConfig) -> torch.Tensor:
    denom = max(cfg.horizon - 1, 1)
    return torch.full_like(active, float(t) / denom)


def neural_policy_inputs(
    history_features: torch.Tensor,
    private_type: torch.Tensor,
    active: torch.Tensor,
    t: int,
    cfg: DynamicConfig,
) -> torch.Tensor:
    time_feature = expand_time_feature(active, t, cfg)
    return torch.cat(
        [
            history_features,
            private_type[:, :, None],
            time_feature[:, :, None],
        ],
        dim=2,
    )


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


def simulate_neural_tagged_policy(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    policy: NeuralReportAuditor,
    exit_threshold: float,
) -> torch.Tensor:
    device = torch.device(cfg.device)
    episodes = draws["tagged_buyer_value"].shape[0]
    threshold_rows = torch.full((episodes,), float(exit_threshold), device=device)

    buyer_values = torch.zeros(episodes, cfg.max_buyers, device=device)
    seller_costs = torch.ones(episodes, cfg.max_sellers, device=device)
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
        buyer_values[:, 0] = draws["tagged_buyer_value"]
        buyer_active[:, 0] = 1.0
        buyer_tag[:, 0] = 1.0
    else:
        seller_costs[:, 0] = draws["tagged_seller_cost"]
        seller_active[:, 0] = 1.0
        seller_tag[:, 0] = 1.0

    utilities_by_episode = torch.zeros(episodes, device=device)
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
                threshold_rows,
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
                threshold_rows,
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
            draws["buyer_arrival"][t],
            draws["buyer_value"][t],
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
            draws["seller_arrival"][t],
            draws["seller_cost"][t],
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
            history_features = deviation_features(
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
            inputs = neural_policy_inputs(history_features, buyer_values, buyer_active, t, cfg)
            tagged_reports = torch.clamp(buyer_values + policy(inputs), 0.0, 1.0)
            buyer_reports = torch.where(buyer_tag > 0.5, tagged_reports, buyer_reports)
        else:
            history_features = deviation_features(
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
            inputs = neural_policy_inputs(history_features, seller_costs, seller_active, t, cfg)
            tagged_reports = torch.clamp(seller_costs + policy(inputs), 0.0, 1.0)
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
        utilities_by_episode = utilities_by_episode + (cfg.discount**t) * tagged_period_utility(
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
        buyer_abandon = ((draws["buyer_abandon"][t] < buyer_abandon_prob) | (buyer_ages >= cfg.max_patience)) & (buyer_active > 0.5) & (~depart_buyers)
        seller_abandon = ((draws["seller_abandon"][t] < seller_abandon_prob) | (seller_ages >= cfg.max_patience)) & (seller_active > 0.5) & (~depart_sellers)
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

    return utilities_by_episode


def regret_summary(regret: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_regret": float(regret.mean().item()),
        f"{prefix}_p95_regret": float(torch.quantile(regret, 0.95).item()),
        f"{prefix}_max_regret": float(regret.max().item()),
        f"{prefix}_regret_count": float(regret.numel()),
    }


def train_policy(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    truthful: torch.Tensor,
    exit_threshold: float,
    hidden_dim: int,
    radius: float,
    steps: int,
    lr: float,
    tail_weight: float,
    tail_alpha: float,
    seed: int,
) -> NeuralReportAuditor:
    torch.manual_seed(seed)
    input_dim = len(FEATURE_NAMES) + 2
    policy = NeuralReportAuditor(input_dim, hidden_dim, radius).to(torch.device(cfg.device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        utility = simulate_neural_tagged_policy(model, cfg, draws, side, policy, exit_threshold)
        gain = utility - truthful
        objective = gain.mean()
        if tail_weight > 0.0 and gain.numel() > 1:
            top_count = max(1, math.ceil((1.0 - tail_alpha) * gain.numel()))
            objective = objective + tail_weight * torch.topk(gain, top_count).values.mean()
        loss = -objective
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
    return policy


def evaluate_policy_family(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    side: str,
    truthful: torch.Tensor,
    policies: list[tuple[NeuralReportAuditor, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    utilities = []
    with torch.no_grad():
        for policy, threshold in policies:
            utilities.append(simulate_neural_tagged_policy(model, cfg, draws, side, policy, threshold))
    stacked = torch.stack(utilities, dim=0)
    best = stacked.max(dim=0).values
    return torch.relu(best - truthful), best


def audit_side(
    model: nn.Module,
    cfg: DynamicConfig,
    side: str,
    train_draws: Dict[str, torch.Tensor],
    val_draws: Dict[str, torch.Tensor],
    test_draws: Dict[str, torch.Tensor],
    exit_thresholds: list[float],
    hidden_dim: int,
    radius: float,
    steps: int,
    lr: float,
    tail_weight: float,
    tail_alpha: float,
    restarts: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    train_truthful = truthful_utility(model, cfg, train_draws, side)
    val_truthful = truthful_utility(model, cfg, val_draws, side)
    test_truthful = truthful_utility(model, cfg, test_draws, side)

    policies: list[tuple[NeuralReportAuditor, float]] = []
    for threshold_index, threshold in enumerate(exit_thresholds):
        for restart in range(restarts):
            policy_seed = seed + 10_000 * threshold_index + 997 * restart
            policy = train_policy(
                model,
                cfg,
                train_draws,
                side,
                train_truthful,
                threshold,
                hidden_dim,
                radius,
                steps,
                lr,
                tail_weight,
                tail_alpha,
                policy_seed,
            )
            policies.append((policy, threshold))

    train_regret, _ = evaluate_policy_family(model, cfg, train_draws, side, train_truthful, policies)
    val_regret, _ = evaluate_policy_family(model, cfg, val_draws, side, val_truthful, policies)
    test_regret, _ = evaluate_policy_family(model, cfg, test_draws, side, test_truthful, policies)
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
                    "learned_continuation_regret": float(value),
                }
            )
    return row, sample_rows


def audit_run(
    run_dir: Path,
    train_episodes: int,
    val_episodes: int,
    test_episodes: int,
    exit_thresholds: list[float],
    hidden_dim: int,
    radius: float,
    steps: int,
    lr: float,
    tail_weight: float,
    tail_alpha: float,
    restarts: int,
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
        exit_thresholds,
        hidden_dim,
        radius,
        steps,
        lr,
        tail_weight,
        tail_alpha,
        restarts,
        cfg.seed + seed_offset + 101,
    )
    seller_row, seller_samples = audit_side(
        model,
        cfg,
        "seller",
        train_draws,
        val_draws,
        test_draws,
        exit_thresholds,
        hidden_dim,
        radius,
        steps,
        lr,
        tail_weight,
        tail_alpha,
        restarts,
        cfg.seed + seed_offset + 503,
    )

    sample_rows = []
    for row in buyer_samples + seller_samples:
        row.update({"run": run_dir.name})
        sample_rows.append(row)

    combined: dict[str, float] = {}
    samples = pd.DataFrame(sample_rows)
    for split in ["train", "val", "test"]:
        split_values = torch.tensor(
            samples.loc[samples["split"] == split, "learned_continuation_regret"].to_numpy(),
            device=device,
            dtype=torch.float32,
        )
        combined.update(regret_summary(split_values, f"learned_{split}"))

    row: dict[str, float | str] = {
        "run": run_dir.name,
        "train_episodes_per_side": float(train_episodes),
        "val_episodes_per_side": float(val_episodes),
        "test_episodes_per_side": float(test_episodes),
        "exit_threshold_count": float(len(exit_thresholds)),
        "restarts": float(restarts),
        "learned_strategy_count_per_side": float(len(exit_thresholds) * restarts),
        "hidden_dim": float(hidden_dim),
        "radius": float(radius),
        "steps": float(steps),
        "lr": float(lr),
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
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--val-episodes", type=int, default=64)
    parser.add_argument("--test-episodes", type=int, default=80)
    parser.add_argument("--exit-thresholds", default="2,4,99")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tail-weight", type=float, default=0.5)
    parser.add_argument("--tail-alpha", type=float, default=0.8)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=300_000)
    parser.add_argument("--seed-stride", type=int, default=997)
    parser.add_argument("--out-dir", default="experiments/learned_continuation_auditor")
    args = parser.parse_args()

    exit_thresholds = [float(value.item()) for value in parse_exit_thresholds(args.exit_thresholds)]
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
        print(f"Training learned continuation auditor for {run_dir.name}", flush=True)
        row, sample_rows = audit_run(
            run_dir,
            args.train_episodes,
            args.val_episodes,
            args.test_episodes,
            exit_thresholds,
            args.hidden_dim,
            args.radius,
            args.steps,
            args.lr,
            args.tail_weight,
            args.tail_alpha,
            args.restarts,
            args.seed_offset + idx * args.seed_stride,
        )
        rows.append(row)
        samples.extend(sample_rows)
        (out_dir / f"{run_dir.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "learned_auditor_by_run.csv", index=False)
    pd.DataFrame(samples).to_csv(out_dir / "learned_auditor_samples.csv", index=False)
    numeric = raw.drop(columns=["run"])
    summary = numeric.agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    summary.to_csv(out_dir / "learned_auditor_summary.csv", index=False)
    print(raw.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
