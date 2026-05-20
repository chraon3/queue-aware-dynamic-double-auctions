from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch

from dynamic_double_auction import (
    ContinuationValueNet,
    DynamicConfig,
    add_arrivals,
    build_dynamic_model,
    compact_queue,
    make_reports,
    mechanism_forward,
    public_queue_features,
    set_seed,
)
from torch import nn

from econpinn_double_auction import utilities


BASE_FEATURE_NAMES = [
    "offset",
    "age",
    "current_imbalance",
    "previous_unmatched",
    "entry_imbalance",
    "average_imbalance",
    "average_queue",
    "unmatched_streak",
]

NONLINEAR_FEATURE_NAMES = [
    "age_sq",
    "abs_current_imbalance",
    "age_x_current_imbalance",
    "age_x_previous_unmatched",
]

PIECEWISE_FEATURE_NAMES = [
    "age_ge_1",
    "age_ge_2",
    "age_ge_3",
    "own_side_excess",
    "own_side_scarce",
    "large_abs_imbalance",
    "high_average_queue",
    "unmatched_streak_positive",
    "unmatched_streak_ge_2",
]

NONLINEAR_HISTORY_FEATURE_NAMES = BASE_FEATURE_NAMES + NONLINEAR_FEATURE_NAMES
PIECEWISE_HISTORY_FEATURE_NAMES = BASE_FEATURE_NAMES + PIECEWISE_FEATURE_NAMES
FEATURE_NAMES = BASE_FEATURE_NAMES + NONLINEAR_FEATURE_NAMES + PIECEWISE_FEATURE_NAMES


def load_config(path: Path) -> DynamicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = DynamicConfig.__dataclass_fields__.keys()
    return DynamicConfig(**{key: value for key, value in data.items() if key in valid})


def load_model(run_dir: Path) -> tuple[DynamicConfig, nn.Module]:
    cfg = load_config(run_dir / "config.json")
    device = torch.device(cfg.device)
    model = build_dynamic_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return cfg, model


def compact_feature(feature: torch.Tensor, active: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    compacted, _, _ = compact_queue(feature, active, feature, fill_value)
    return compacted


def make_grid_candidates(
    radius: torch.Tensor,
    feature_names: Iterable[str],
    device: torch.device,
) -> torch.Tensor:
    feature_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    names = list(feature_names)
    candidates = []
    for coeffs in itertools.product(radius.tolist(), repeat=len(names)):
        row = torch.zeros(len(FEATURE_NAMES), device=device)
        for name, coeff in zip(names, coeffs):
            row[feature_index[name]] = float(coeff)
        candidates.append(row)
    return torch.stack(candidates, dim=0)


def make_history_candidates(
    radius_value: float,
    grid: int,
    draws: int,
    seed: int,
    device: torch.device,
    feature_names: Iterable[str] = BASE_FEATURE_NAMES,
) -> torch.Tensor:
    radius = torch.linspace(-radius_value, radius_value, grid, device=device)
    state = make_grid_candidates(
        radius,
        ["offset", "age", "current_imbalance", "previous_unmatched"],
        device,
    )
    if draws <= 0:
        return state
    names = list(feature_names)
    feature_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    engine = torch.quasirandom.SobolEngine(len(names), scramble=True, seed=seed)
    sobol = engine.draw(draws).to(device)
    history = torch.zeros(draws, len(FEATURE_NAMES), device=device)
    scaled = (2.0 * sobol - 1.0) * radius_value
    for column, feature_name in enumerate(names):
        history[:, feature_index[feature_name]] = scaled[:, column]
    all_candidates = torch.cat([state, history], dim=0)
    return torch.unique(torch.round(all_candidates * 1.0e6) / 1.0e6, dim=0)


def candidate_matrix(
    family: str,
    radius_value: float,
    grid: int,
    history_draws: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    radius = torch.linspace(-radius_value, radius_value, grid, device=device)
    if family == "age":
        return make_grid_candidates(radius, ["offset", "age"], device)
    if family == "state":
        return make_grid_candidates(radius, ["offset", "age", "current_imbalance", "previous_unmatched"], device)
    if family == "history":
        return make_history_candidates(radius_value, grid, history_draws, seed, device, BASE_FEATURE_NAMES)
    if family == "history_nonlinear":
        return make_history_candidates(radius_value, grid, history_draws, seed, device, NONLINEAR_HISTORY_FEATURE_NAMES)
    if family == "history_piecewise":
        return make_history_candidates(radius_value, grid, history_draws, seed, device, PIECEWISE_HISTORY_FEATURE_NAMES)
    raise ValueError(f"Unknown audit family: {family}")


def deviation_features(
    active: torch.Tensor,
    ages: torch.Tensor,
    prev_unmatched: torch.Tensor,
    entry_imbalance: torch.Tensor,
    imbalance_exposure: torch.Tensor,
    queue_exposure: torch.Tensor,
    unmatched_streak: torch.Tensor,
    side_imbalance: torch.Tensor,
    cfg: DynamicConfig,
) -> torch.Tensor:
    age_scale = torch.clamp(ages, min=0.0) / max(cfg.max_patience, 1)
    history_denominator = torch.clamp(ages, min=1.0)
    average_imbalance = imbalance_exposure / history_denominator
    average_queue = queue_exposure / history_denominator
    unmatched_rate = unmatched_streak / max(cfg.max_patience, 1)
    current_imbalance = side_imbalance[:, None].expand_as(active)
    features = torch.stack(
        [
            torch.ones_like(active),
            age_scale,
            current_imbalance,
            prev_unmatched,
            entry_imbalance,
            average_imbalance,
            average_queue,
            unmatched_rate,
            age_scale.pow(2.0),
            torch.abs(current_imbalance),
            age_scale * current_imbalance,
            age_scale * prev_unmatched,
            (ages >= 1.0).float(),
            (ages >= 2.0).float(),
            (ages >= 3.0).float(),
            (current_imbalance > 0.0).float(),
            (current_imbalance < 0.0).float(),
            (torch.abs(current_imbalance) >= 0.25).float(),
            (average_queue >= 0.5).float(),
            (unmatched_streak >= 1.0).float(),
            (unmatched_streak >= 2.0).float(),
        ],
        dim=2,
    )
    return torch.where(active[:, :, None] > 0.5, features, torch.zeros_like(features))


def update_best_deviation(
    model: nn.Module,
    buyer_reports: torch.Tensor,
    seller_reports: torch.Tensor,
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    deviations: torch.Tensor,
    is_buyer: bool,
    chunk_size: int,
    public_state: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = buyer_reports.shape[0]
    width = buyer_reports.shape[1] if is_buyer else seller_reports.shape[1]
    best = torch.full((batch_size, width), -1.0e9, device=buyer_reports.device)
    for start in range(0, deviations.shape[0], chunk_size):
        end = min(start + chunk_size, deviations.shape[0])
        chunk = deviations[start:end]
        chunk_count = end - start
        expanded_buyers = buyer_reports[None, :, :].expand(chunk_count, -1, -1).reshape(chunk_count * batch_size, -1)
        expanded_sellers = seller_reports[None, :, :].expand(chunk_count, -1, -1).reshape(chunk_count * batch_size, -1)
        expanded_values = buyer_values[None, :, :].expand(chunk_count, -1, -1).reshape(chunk_count * batch_size, -1)
        expanded_costs = seller_costs[None, :, :].expand(chunk_count, -1, -1).reshape(chunk_count * batch_size, -1)
        for agent_idx in range(width):
            b_rep = expanded_buyers.clone()
            s_rep = expanded_sellers.clone()
            if is_buyer:
                b_rep[:, agent_idx] = chunk[:, :, agent_idx].reshape(-1)
            else:
                s_rep[:, agent_idx] = chunk[:, :, agent_idx].reshape(-1)
            out = mechanism_forward(model, b_rep, s_rep, public_state)
            u_b, u_s = utilities(out, expanded_values, expanded_costs)
            utility = u_b[:, agent_idx] if is_buyer else u_s[:, agent_idx]
            utility = utility.reshape(chunk_count, batch_size).max(dim=0).values
            best[:, agent_idx] = torch.maximum(best[:, agent_idx], utility)
    return best


def history_conditioned_regret(
    model: nn.Module,
    cfg: DynamicConfig,
    batch_size: int,
    candidates: torch.Tensor,
    chunk_size: int,
) -> Dict[str, torch.Tensor]:
    device = torch.device(cfg.device)
    buyer_values = torch.zeros(batch_size, cfg.max_buyers, device=device)
    seller_costs = torch.ones(batch_size, cfg.max_sellers, device=device)
    buyer_active = torch.zeros_like(buyer_values)
    seller_active = torch.zeros_like(seller_costs)
    buyer_ages = torch.zeros_like(buyer_values)
    seller_ages = torch.zeros_like(seller_costs)
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
    regret_samples = []

    for _t in range(cfg.horizon):
        buyer_prev_unmatched = compact_feature(buyer_prev_unmatched, buyer_active)
        seller_prev_unmatched = compact_feature(seller_prev_unmatched, seller_active)
        buyer_entry_imbalance = compact_feature(buyer_entry_imbalance, buyer_active)
        seller_entry_imbalance = compact_feature(seller_entry_imbalance, seller_active)
        buyer_imbalance_exposure = compact_feature(buyer_imbalance_exposure, buyer_active)
        seller_imbalance_exposure = compact_feature(seller_imbalance_exposure, seller_active)
        buyer_queue_exposure = compact_feature(buyer_queue_exposure, buyer_active)
        seller_queue_exposure = compact_feature(seller_queue_exposure, seller_active)
        buyer_unmatched_streak = compact_feature(buyer_unmatched_streak, buyer_active)
        seller_unmatched_streak = compact_feature(seller_unmatched_streak, seller_active)
        buyer_values, buyer_active, buyer_ages = compact_queue(buyer_values, buyer_active, buyer_ages, 0.0)
        seller_costs, seller_active, seller_ages = compact_queue(seller_costs, seller_active, seller_ages, 1.0)
        pre_buyer_active = buyer_active.clone()
        pre_seller_active = seller_active.clone()
        buyer_values, buyer_active, buyer_ages = add_arrivals(buyer_values, buyer_active, buyer_ages, cfg.arrival_prob_buyer, 0.0, True)
        seller_costs, seller_active, seller_ages = add_arrivals(seller_costs, seller_active, seller_ages, cfg.arrival_prob_seller, 1.0, False)
        new_buyers = (buyer_active > 0.5) & (pre_buyer_active < 0.5)
        new_sellers = (seller_active > 0.5) & (pre_seller_active < 0.5)
        buyer_count = buyer_active.sum(dim=1)
        seller_count = seller_active.sum(dim=1)
        total_capacity = max(cfg.max_buyers + cfg.max_sellers, 1)
        current_imbalance = (buyer_count - seller_count) / total_capacity
        current_queue = (buyer_count + seller_count) / total_capacity
        buyer_side_imbalance = current_imbalance
        seller_side_imbalance = -current_imbalance
        buyer_prev_unmatched = torch.where(buyer_active > 0.5, buyer_prev_unmatched, torch.zeros_like(buyer_prev_unmatched))
        seller_prev_unmatched = torch.where(seller_active > 0.5, seller_prev_unmatched, torch.zeros_like(seller_prev_unmatched))
        buyer_entry_imbalance = torch.where(new_buyers, buyer_side_imbalance[:, None], buyer_entry_imbalance)
        seller_entry_imbalance = torch.where(new_sellers, seller_side_imbalance[:, None], seller_entry_imbalance)

        buyer_reports, seller_reports = make_reports(buyer_values, seller_costs, buyer_active, seller_active)
        public_state = public_queue_features(buyer_active, seller_active, buyer_ages, seller_ages, _t, cfg)
        truthful = mechanism_forward(model, buyer_reports, seller_reports, public_state)
        truthful_buyer_u, truthful_seller_u = utilities(truthful, buyer_values, seller_costs)

        buyer_features = deviation_features(
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
        seller_features = deviation_features(
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
        buyer_delta = torch.einsum("kd,bnd->kbn", candidates, buyer_features)
        seller_delta = torch.einsum("kd,bnd->kbn", candidates, seller_features)
        buyer_deviations = torch.clamp(buyer_values[None, :, :] + buyer_delta, 0.0, 1.0)
        seller_deviations = torch.clamp(seller_costs[None, :, :] + seller_delta, 0.0, 1.0)
        best_buyer = update_best_deviation(
            model,
            buyer_reports,
            seller_reports,
            buyer_values,
            seller_costs,
            buyer_deviations,
            True,
            chunk_size,
            public_state,
        )
        best_seller = update_best_deviation(
            model,
            buyer_reports,
            seller_reports,
            buyer_values,
            seller_costs,
            seller_deviations,
            False,
            chunk_size,
            public_state,
        )

        buyer_regret = torch.relu(best_buyer - truthful_buyer_u) * buyer_active
        seller_regret = torch.relu(best_seller - truthful_seller_u) * seller_active
        if torch.any(buyer_active > 0.5):
            regret_samples.append(buyer_regret[buyer_active > 0.5])
        if torch.any(seller_active > 0.5):
            regret_samples.append(seller_regret[seller_active > 0.5])

        out = truthful
        match = out["match"] * (buyer_active[:, :, None] * seller_active[:, None, :])
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
        buyer_abandon = ((torch.rand_like(buyer_active) < buyer_abandon_prob) | (buyer_ages >= cfg.max_patience)) & (buyer_active > 0.5) & (~depart_buyers)
        seller_abandon = ((torch.rand_like(seller_active) < seller_abandon_prob) | (seller_ages >= cfg.max_patience)) & (seller_active > 0.5) & (~depart_sellers)
        surviving_buyers = (buyer_active > 0.5) & (~depart_buyers) & (~buyer_abandon)
        surviving_sellers = (seller_active > 0.5) & (~depart_sellers) & (~seller_abandon)
        buyer_imbalance_exposure = torch.where(
            surviving_buyers,
            buyer_imbalance_exposure + buyer_side_imbalance[:, None],
            torch.zeros_like(buyer_imbalance_exposure),
        )
        seller_imbalance_exposure = torch.where(
            surviving_sellers,
            seller_imbalance_exposure + seller_side_imbalance[:, None],
            torch.zeros_like(seller_imbalance_exposure),
        )
        buyer_queue_exposure = torch.where(
            surviving_buyers,
            buyer_queue_exposure + current_queue[:, None],
            torch.zeros_like(buyer_queue_exposure),
        )
        seller_queue_exposure = torch.where(
            surviving_sellers,
            seller_queue_exposure + current_queue[:, None],
            torch.zeros_like(seller_queue_exposure),
        )
        buyer_unmatched_streak = torch.where(
            surviving_buyers,
            buyer_unmatched_streak + (unmatched_buyers.detach() > 0.5).float(),
            torch.zeros_like(buyer_unmatched_streak),
        )
        seller_unmatched_streak = torch.where(
            surviving_sellers,
            seller_unmatched_streak + (unmatched_sellers.detach() > 0.5).float(),
            torch.zeros_like(seller_unmatched_streak),
        )
        buyer_active = torch.where(depart_buyers | buyer_abandon, torch.zeros_like(buyer_active), buyer_active)
        seller_active = torch.where(depart_sellers | seller_abandon, torch.zeros_like(seller_active), seller_active)
        buyer_prev_unmatched = torch.where(buyer_active > 0.5, unmatched_buyers.detach(), torch.zeros_like(buyer_prev_unmatched))
        seller_prev_unmatched = torch.where(seller_active > 0.5, unmatched_sellers.detach(), torch.zeros_like(seller_prev_unmatched))
        buyer_values = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
        seller_costs = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
        buyer_ages = torch.where(buyer_active > 0.5, buyer_ages, torch.zeros_like(buyer_ages))
        seller_ages = torch.where(seller_active > 0.5, seller_ages, torch.zeros_like(seller_ages))
        buyer_entry_imbalance = torch.where(buyer_active > 0.5, buyer_entry_imbalance, torch.zeros_like(buyer_entry_imbalance))
        seller_entry_imbalance = torch.where(seller_active > 0.5, seller_entry_imbalance, torch.zeros_like(seller_entry_imbalance))

    regrets = torch.cat([sample.reshape(-1) for sample in regret_samples])
    return {
        "history_mean_regret": regrets.mean(),
        "history_p95_regret": torch.quantile(regrets, 0.95),
        "history_max_regret": regrets.max(),
        "history_regret_count": torch.tensor(float(regrets.numel()), device=device),
    }


def audit_run(
    run_dir: Path,
    episodes: int,
    grid: int,
    radius_value: float,
    family: str,
    history_draws: int,
    chunk_size: int,
    seed_offset: int,
) -> Dict[str, float | str]:
    cfg, model = load_model(run_dir)
    cfg = replace(cfg, eval_episodes=episodes)
    set_seed(cfg.seed + seed_offset)
    device = torch.device(cfg.device)
    candidates = candidate_matrix(family, radius_value, grid, history_draws, cfg.seed + seed_offset, device)
    with torch.no_grad():
        audit = history_conditioned_regret(model, cfg, episodes, candidates, chunk_size)
    row: Dict[str, float | str] = {
        "run": run_dir.name,
        "audit_family": family,
        "episodes": float(episodes),
        "grid": float(grid),
        "radius": float(radius_value),
        "candidate_count": float(candidates.shape[0]),
        "history_draws": float(history_draws if family.startswith("history") else 0),
    }
    row.update({key: float(value.item()) for key, value in audit.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--family", choices=["age", "state", "history", "history_nonlinear", "history_piecewise"], default="history")
    parser.add_argument("--history-draws", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--out-dir", default="experiments/dynamic_history_audit")
    args = parser.parse_args()

    run_dirs = []
    for pattern in args.runs:
        run_dirs.extend(sorted(Path().glob(pattern)))
    run_dirs = [path for path in run_dirs if (path / "model.pt").exists()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, run_dir in enumerate(run_dirs):
        print(f"Auditing {run_dir.name}", flush=True)
        row = audit_run(
            run_dir,
            args.episodes,
            args.grid,
            args.radius,
            args.family,
            args.history_draws,
            args.chunk_size,
            50_000 + idx * 997,
        )
        rows.append(row)
        (out_dir / f"{run_dir.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "history_audit_by_run.csv", index=False)
    numeric = raw.drop(columns=["run", "audit_family"])
    summary = numeric.agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    summary.to_csv(out_dir / "history_audit_summary.csv", index=False)
    print(raw.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
