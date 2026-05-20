from __future__ import annotations

import argparse
import csv
import copy
import json
from dataclasses import asdict, replace
from collections import defaultdict
from pathlib import Path

import torch

from dynamic_continuation_audit import (
    build_strategy_set,
    continuation_regret_components_for_side,
    continuation_regret_for_side,
    make_base_draws,
    parse_exit_thresholds,
)
from dynamic_double_auction import DynamicConfig, build_dynamic_model, evaluate_dynamic_baselines, evaluate_static_baselines, simulate_dynamic
from dynamic_history_audit import FEATURE_NAMES, load_config
from torch import nn


def top_tail_mean(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of the largest (1-alpha) share of a regret vector."""
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.new_tensor(0.0)
    alpha = min(max(alpha, 0.0), 0.999)
    k = max(1, int(round((1.0 - alpha) * flat.numel())))
    return torch.topk(flat, k=k, largest=True).values.mean()


def baseline_excess_stats(
    samples: torch.Tensor | None,
    baseline_samples: torch.Tensor | None,
    alpha: float,
    tail_weight: float,
    max_weight: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if samples is None:
        zero = torch.tensor(0.0)
        return zero, zero, zero, zero, 0
    zero = samples.new_tensor(0.0)
    if baseline_samples is None or samples.numel() == 0:
        return zero, zero, zero, zero, 0
    if samples.shape != baseline_samples.shape:
        raise ValueError(f"Candidate and baseline regret samples differ in shape: {samples.shape} vs {baseline_samples.shape}")
    excess = torch.relu(samples - baseline_samples.detach() - margin)
    mean = excess.mean()
    tail = top_tail_mean(excess, alpha)
    max_value = excess.max()
    loss = mean + tail_weight * tail + max_weight * max_value
    return loss, mean, tail, max_value, int(excess.numel())


def select_draw_episodes(draws: dict[str, torch.Tensor], episodes: list[int]) -> dict[str, torch.Tensor]:
    index = torch.tensor(episodes, device=next(iter(draws.values())).device, dtype=torch.long)
    selected: dict[str, torch.Tensor] = {}
    for key, value in draws.items():
        dim = 0 if key.startswith("tagged_") else 1
        selected[key] = torch.index_select(value, dim, index)
    return selected


def concat_draw_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor] | None:
    if not batches:
        return None
    combined: dict[str, torch.Tensor] = {}
    for key in batches[0]:
        dim = 0 if key.startswith("tagged_") else 1
        combined[key] = torch.cat([batch[key] for batch in batches], dim=dim)
    return combined


def append_strategy_rows(
    base_coefficients: torch.Tensor,
    base_exit_thresholds: torch.Tensor,
    extra_coefficients: torch.Tensor | None,
    extra_exit_thresholds: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if extra_coefficients is None or extra_exit_thresholds is None or extra_coefficients.numel() == 0:
        return base_coefficients, base_exit_thresholds
    return (
        torch.cat([base_coefficients, extra_coefficients.to(base_coefficients.device)], dim=0),
        torch.cat([base_exit_thresholds, extra_exit_thresholds.to(base_exit_thresholds.device)], dim=0),
    )


def load_hard_case_bank(
    csv_path: str,
    cfg: DynamicConfig,
    device: torch.device,
    run_name: str,
    top_k: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor | None, torch.Tensor | None, dict[str, object]]:
    if not csv_path:
        return {}, None, None, {"path": "", "case_count": 0}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Hard-case CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, None, None, {"path": str(path), "case_count": 0}
    matching_rows = [row for row in rows if row.get("run") == run_name]
    if matching_rows:
        rows = matching_rows
    rows = sorted(rows, key=lambda row: float(row.get("regret") or 0.0), reverse=True)
    if top_k > 0:
        rows = rows[:top_k]

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    strategy_rows: list[list[float]] = []
    strategy_exits: list[float] = []
    seen_strategies: set[tuple[float, ...]] = set()
    for row in rows:
        side = str(row.get("side", "")).strip().lower()
        if side not in {"buyer", "seller"}:
            continue
        episode = int(float(row.get("episode", 0.0)))
        seed_offset = int(float(row.get("seed_offset", 80_000.0)))
        grouped[(side, seed_offset)].append(episode)
        coefficient_values: list[float] = []
        for feature_name in FEATURE_NAMES:
            key = f"coef_{feature_name}"
            if key not in row or row[key] in {"", None}:
                coefficient_values.append(0.0)
            else:
                coefficient_values.append(float(row[key]))
        if row.get("exit_threshold") not in {"", None}:
            threshold = float(row["exit_threshold"])
            strategy_key = tuple(round(value, 8) for value in coefficient_values + [threshold])
            if strategy_key not in seen_strategies:
                seen_strategies.add(strategy_key)
                strategy_rows.append(coefficient_values)
                strategy_exits.append(threshold)

    side_batches: dict[str, list[dict[str, torch.Tensor]]] = {"buyer": [], "seller": []}
    for (side, seed_offset), episodes in grouped.items():
        unique_episodes = sorted(set(episodes))
        if not unique_episodes:
            continue
        draw_seed = cfg.seed + seed_offset + (11 if side == "buyer" else 29)
        full_draws = make_base_draws(cfg, max(unique_episodes) + 1, draw_seed, device)
        side_batches[side].append(select_draw_episodes(full_draws, unique_episodes))

    draws_by_side = {
        side: draws
        for side, draws in (
            ("buyer", concat_draw_batches(side_batches["buyer"])),
            ("seller", concat_draw_batches(side_batches["seller"])),
        )
        if draws is not None
    }
    if strategy_rows:
        coefficients = torch.tensor(strategy_rows, device=device, dtype=torch.float32)
        exit_thresholds = torch.tensor(strategy_exits, device=device, dtype=torch.float32)
    else:
        coefficients = None
        exit_thresholds = None
    manifest = {
        "path": str(path),
        "case_count": sum(draws["tagged_buyer_value"].shape[0] if side == "buyer" else draws["tagged_seller_cost"].shape[0] for side, draws in draws_by_side.items()),
        "buyer_cases": int(draws_by_side.get("buyer", {}).get("tagged_buyer_value", torch.empty(0, device=device)).shape[0]) if "buyer" in draws_by_side else 0,
        "seller_cases": int(draws_by_side.get("seller", {}).get("tagged_seller_cost", torch.empty(0, device=device)).shape[0]) if "seller" in draws_by_side else 0,
        "strategy_count": len(strategy_rows),
        "top_k": top_k,
    }
    return draws_by_side, coefficients, exit_thresholds, manifest


def continuation_samples_from_bank(
    model: nn.Module,
    cfg: DynamicConfig,
    draws_by_side: dict[str, dict[str, torch.Tensor]],
    coefficients: torch.Tensor,
    exit_thresholds: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor | None:
    samples = []
    for side in ("buyer", "seller"):
        draws = draws_by_side.get(side)
        if draws is None:
            continue
        regret, _ = continuation_regret_for_side(model, cfg, draws, side, coefficients, exit_thresholds, chunk_size)
        samples.append(regret)
    if not samples:
        return None
    return torch.cat(samples)


def unique_strategy_rows(
    coefficients: torch.Tensor | None,
    exit_thresholds: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if coefficients is None or exit_thresholds is None or coefficients.numel() == 0:
        return coefficients, exit_thresholds
    seen: set[tuple[float, ...]] = set()
    keep: list[int] = []
    for idx in range(coefficients.shape[0]):
        row = coefficients[idx].detach().cpu().tolist()
        threshold = float(exit_thresholds[idx].detach().cpu().item())
        key = tuple(round(float(value), 7) for value in row + [threshold])
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    if len(keep) == coefficients.shape[0]:
        return coefficients, exit_thresholds
    index = torch.tensor(keep, device=coefficients.device, dtype=torch.long)
    return coefficients.index_select(0, index), exit_thresholds.index_select(0, index)


def mine_online_hard_cases(
    model: nn.Module,
    cfg: DynamicConfig,
    mining_coefficients: torch.Tensor,
    mining_exit_thresholds: torch.Tensor,
    episodes: int,
    top_k: int,
    seed: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor | None, torch.Tensor | None, dict[str, object]]:
    """Mine a broad, current-model hard pool without backpropagating through the mining pass.

    The selected pool is deliberately distributional rather than a tiny replay file:
    it re-draws fresh continuation episodes and a wider report-strategy family, then
    keeps the worst episodes and the strategies that attained them.
    """
    if episodes <= 0 or top_k <= 0:
        return {}, None, None, {"enabled": False, "case_count": 0}

    draws_by_side: dict[str, dict[str, torch.Tensor]] = {}
    strategy_rows: list[torch.Tensor] = []
    strategy_exits: list[torch.Tensor] = []
    side_summaries: dict[str, object] = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for side, seed_shift in (("buyer", 11), ("seller", 29)):
            draws = make_base_draws(cfg, episodes, seed + seed_shift, device)
            regret, _no_exit_regret, _truthful, _best_values, best_indices = continuation_regret_components_for_side(
                model,
                cfg,
                draws,
                side,
                mining_coefficients,
                mining_exit_thresholds,
                chunk_size,
            )
            if regret.numel() == 0:
                continue
            keep_count = min(top_k, regret.numel())
            top_values, top_episodes = torch.topk(regret.detach(), keep_count)
            episode_indices = [int(idx) for idx in top_episodes.detach().cpu().tolist()]
            draws_by_side[side] = select_draw_episodes(draws, episode_indices)
            selected_strategies = best_indices.index_select(0, top_episodes).detach().long()
            strategy_rows.append(mining_coefficients.index_select(0, selected_strategies))
            strategy_exits.append(mining_exit_thresholds.index_select(0, selected_strategies))
            side_summaries[side] = {
                "episodes_mined": int(episodes),
                "cases_kept": int(keep_count),
                "mean_regret": float(regret.mean().item()),
                "tail_mean_regret": float(top_values.mean().item()),
                "max_regret": float(top_values.max().item()),
                "selected_episodes": episode_indices,
            }
    if was_training:
        model.train()
    if strategy_rows:
        coefficients, exit_thresholds = unique_strategy_rows(torch.cat(strategy_rows, dim=0), torch.cat(strategy_exits, dim=0))
    else:
        coefficients, exit_thresholds = None, None
    case_count = sum(int(draws["tagged_buyer_value"].shape[0]) for draws in draws_by_side.values())
    manifest = {
        "enabled": True,
        "seed": int(seed),
        "episodes_per_side": int(episodes),
        "top_k_per_side": int(top_k),
        "case_count": int(case_count),
        "strategy_count": int(coefficients.shape[0]) if coefficients is not None else 0,
        "mining_strategy_count": int(mining_coefficients.shape[0]),
        "side_summaries": side_summaries,
    }
    return draws_by_side, coefficients, exit_thresholds, manifest


def load_base(run_dir: Path) -> tuple[DynamicConfig, nn.Module]:
    cfg = load_config(run_dir / "config.json")
    device = torch.device(cfg.device)
    model = build_dynamic_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return cfg, model


def evaluate_and_save(model: nn.Module, cfg: DynamicConfig, out_dir: Path) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        sim = simulate_dynamic(model, cfg, cfg.eval_episodes, train=True)
    metrics = {key: float(value.item()) for key, value in sim.items()}
    metrics.update(evaluate_static_baselines(cfg))
    metrics.update(evaluate_dynamic_baselines(cfg))
    first_best_obj = max(metrics["dynamic_first_best_objective"], 1.0e-9)
    metrics["dynamic_neural_efficiency"] = metrics["objective"] / first_best_obj
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_first_best_objective(run_dir: Path) -> float:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return 1.0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(max(metrics.get("dynamic_first_best_objective", 1.0), 1.0e-9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=7.0e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cont-episodes", type=int, default=24)
    parser.add_argument("--cont-weight", type=float, default=0.8)
    parser.add_argument("--cont-tail-weight", type=float, default=0.0)
    parser.add_argument("--cont-max-weight", type=float, default=0.0)
    parser.add_argument("--hard-case-csv", default="", help="Optional continuation_audit_worst_cases.csv file for hard-case replay training.")
    parser.add_argument("--hard-case-top-k", type=int, default=0, help="Number of worst cases to replay from --hard-case-csv; 0 uses all rows after run filtering.")
    parser.add_argument("--hard-case-weight", type=float, default=0.0)
    parser.add_argument("--hard-case-tail-weight", type=float, default=0.0)
    parser.add_argument("--hard-case-max-weight", type=float, default=0.0)
    parser.add_argument("--validation-hard-case-csv", default="", help="Optional held-out hard-case CSV used only for checkpoint selection.")
    parser.add_argument("--selection-hard-case-weight", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0, help="L2 trust-region penalty around the loaded base mechanism parameters.")
    parser.add_argument("--online-hard-weight", type=float, default=0.0, help="Weight on a refreshed online hard-continuation pool.")
    parser.add_argument("--online-hard-tail-weight", type=float, default=0.0)
    parser.add_argument("--online-hard-max-weight", type=float, default=0.0)
    parser.add_argument("--online-hard-refresh-every", type=int, default=0, help="Refresh online hard cases every N steps; 0 disables mining.")
    parser.add_argument("--online-hard-episodes", type=int, default=0, help="Fresh episodes per side used for online hard-case mining.")
    parser.add_argument("--online-hard-top-k", type=int, default=0, help="Worst episodes per side kept after each online mining pass.")
    parser.add_argument("--online-hard-history-draws", type=int, default=0, help="Sobol history strategies used by the online mining pass; 0 falls back to --history-draws.")
    parser.add_argument("--online-hard-grid", type=int, default=0, help="Grid used by online mining; 0 falls back to --grid.")
    parser.add_argument("--online-hard-radius", type=float, default=0.45)
    parser.add_argument("--online-hard-seed-offset", type=int, default=700_000)
    parser.add_argument("--selection-online-hard-weight", type=float, default=0.0)
    parser.add_argument("--validation-online-hard-episodes", type=int, default=0, help="Held-out online hard mining episodes for checkpoint selection; 0 falls back to --validation-episodes.")
    parser.add_argument("--fixed-validation-online-hard", action="store_true", help="Mine one held-out online hard pool from the base model and reuse it for all checkpoint selection scores.")
    parser.add_argument("--tail-alpha", type=float, default=0.8)
    parser.add_argument("--p95-regret-weight", type=float, default=0.0)
    parser.add_argument("--p95-regret-target", type=float, default=0.08)
    parser.add_argument("--refresh-every", type=int, default=0)
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--include-baseline-selection", action="store_true", help="Treat the loaded base mechanism as a step-0 checkpoint candidate.")
    parser.add_argument("--baseline-selection-margin", type=float, default=0.0, help="Require a fine-tuned checkpoint to beat the baseline validation score by this margin.")
    parser.add_argument("--baseline-delta-weight", type=float, default=0.0, help="Training penalty on paired continuation-regret deterioration relative to the loaded base mechanism.")
    parser.add_argument("--baseline-delta-tail-weight", type=float, default=1.0)
    parser.add_argument("--baseline-delta-max-weight", type=float, default=1.0)
    parser.add_argument("--baseline-delta-margin", type=float, default=0.0)
    parser.add_argument("--baseline-delta-episodes", type=int, default=0, help="Extra paired episodes per side reserved for baseline-delta training; 0 reuses the continuation batch.")
    parser.add_argument("--baseline-delta-seed-offset", type=int, default=2_100_000)
    parser.add_argument("--selection-baseline-delta-weight", type=float, default=0.0, help="Checkpoint-selection penalty on paired continuation-regret deterioration relative to the loaded base mechanism.")
    parser.add_argument("--validation-every", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--validation-objective-episodes", type=int, default=0)
    parser.add_argument("--selection-objective-weight", type=float, default=0.2)
    parser.add_argument("--selection-max-weight", type=float, default=0.0)
    parser.add_argument("--selection-efficiency-target", type=float, default=0.0)
    parser.add_argument("--selection-efficiency-weight", type=float, default=0.0)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--history-draws", type=int, default=8)
    parser.add_argument("--exit-thresholds", default="2,99")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=450)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, model = load_base(run_dir)
    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    for parameter in baseline_model.parameters():
        parameter.requires_grad_(False)
    anchor_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    seed = args.seed or cfg.seed + 900_000
    first_best_objective = load_first_best_objective(run_dir)
    cfg = replace(
        cfg,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        out_dir=str(out_dir),
        value_loss_weight=0.0,
        congestion_aux_weight=0.0,
        imitation_aux_weight=0.0,
    )
    torch.manual_seed(seed)
    device = torch.device(cfg.device)
    exit_thresholds = parse_exit_thresholds(args.exit_thresholds)
    def make_strategies(strategy_seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return build_strategy_set(
            "history",
            0.35,
            args.grid,
            args.history_draws,
            exit_thresholds,
            strategy_seed,
            device,
        )

    coefficients, strategy_exit_thresholds, report_count = make_strategies(seed)
    validation_coefficients, validation_exit_thresholds, _ = make_strategies(seed + 555_001)
    online_hard_enabled = (
        args.online_hard_refresh_every > 0
        and args.online_hard_episodes > 0
        and args.online_hard_top_k > 0
        and (args.online_hard_weight > 0.0 or args.selection_online_hard_weight > 0.0)
    )
    online_mining_coefficients: torch.Tensor | None = None
    online_mining_exit_thresholds: torch.Tensor | None = None
    online_mining_report_count = 0
    validation_online_mining_coefficients: torch.Tensor | None = None
    validation_online_mining_exit_thresholds: torch.Tensor | None = None
    if online_hard_enabled:
        online_grid = args.online_hard_grid or args.grid
        online_history_draws = args.online_hard_history_draws or args.history_draws
        online_mining_coefficients, online_mining_exit_thresholds, online_mining_report_count = build_strategy_set(
            "history",
            args.online_hard_radius,
            online_grid,
            online_history_draws,
            exit_thresholds,
            seed + args.online_hard_seed_offset,
            device,
        )
        validation_online_mining_coefficients, validation_online_mining_exit_thresholds, _ = build_strategy_set(
            "history",
            args.online_hard_radius,
            online_grid,
            online_history_draws,
            exit_thresholds,
            seed + args.online_hard_seed_offset + 555_001,
            device,
        )
    hard_case_draws, hard_case_extra_coefficients, hard_case_extra_exits, hard_case_manifest = load_hard_case_bank(
        args.hard_case_csv,
        cfg,
        device,
        run_dir.name,
        args.hard_case_top_k,
    )
    hard_case_coefficients, hard_case_exit_thresholds = append_strategy_rows(
        coefficients,
        strategy_exit_thresholds,
        hard_case_extra_coefficients,
        hard_case_extra_exits,
    )
    validation_hard_case_draws, validation_hard_case_extra_coefficients, validation_hard_case_extra_exits, validation_hard_case_manifest = load_hard_case_bank(
        args.validation_hard_case_csv,
        cfg,
        device,
        run_dir.name,
        args.hard_case_top_k,
    )
    if not validation_hard_case_draws:
        validation_hard_case_draws = hard_case_draws
        validation_hard_case_extra_coefficients = hard_case_extra_coefficients
        validation_hard_case_extra_exits = hard_case_extra_exits
        validation_hard_case_manifest = dict(hard_case_manifest)
        validation_hard_case_manifest["fallback_to_training_bank"] = True
    validation_hard_case_coefficients, validation_hard_case_exit_thresholds = append_strategy_rows(
        validation_coefficients,
        validation_exit_thresholds,
        validation_hard_case_extra_coefficients,
        validation_hard_case_extra_exits,
    )
    validation_buyer_draws = make_base_draws(cfg, args.validation_episodes, seed + 555_011, device)
    validation_seller_draws = make_base_draws(cfg, args.validation_episodes, seed + 555_029, device)
    fixed_validation_online_draws: dict[str, dict[str, torch.Tensor]] = {}
    fixed_validation_online_coefficients = validation_coefficients
    fixed_validation_online_exit_thresholds = validation_exit_thresholds
    fixed_validation_online_manifest: dict[str, object] = {"enabled": False}
    if args.fixed_validation_online_hard and args.selection_online_hard_weight > 0.0 and online_hard_enabled:
        assert validation_online_mining_coefficients is not None
        assert validation_online_mining_exit_thresholds is not None
        fixed_validation_online_draws, fixed_validation_online_extra_coefficients, fixed_validation_online_extra_exits, fixed_validation_online_manifest = mine_online_hard_cases(
            model,
            cfg,
            validation_online_mining_coefficients,
            validation_online_mining_exit_thresholds,
            args.validation_online_hard_episodes or args.validation_episodes,
            args.online_hard_top_k,
            seed + args.online_hard_seed_offset + 888_000,
            device,
            args.chunk_size,
        )
        fixed_validation_online_manifest["fixed_for_selection"] = True
        fixed_validation_online_coefficients, fixed_validation_online_exit_thresholds = append_strategy_rows(
            validation_coefficients,
            validation_exit_thresholds,
            fixed_validation_online_extra_coefficients,
            fixed_validation_online_extra_exits,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def compute_selection_row(step: int) -> dict[str, float | int]:
        model.eval()
        with torch.no_grad():
            objective_episodes = args.validation_objective_episodes or min(cfg.batch_size, args.eval_episodes)
            val_sim = simulate_dynamic(model, cfg, objective_episodes, train=True)
            val_buyer_regret, _ = continuation_regret_for_side(
                model,
                cfg,
                validation_buyer_draws,
                "buyer",
                validation_coefficients,
                validation_exit_thresholds,
                args.chunk_size,
            )
            val_seller_regret, _ = continuation_regret_for_side(
                model,
                cfg,
                validation_seller_draws,
                "seller",
                validation_coefficients,
                validation_exit_thresholds,
                args.chunk_size,
            )
            val_continuation = torch.cat([val_buyer_regret, val_seller_regret])
            val_mean = val_continuation.mean()
            val_tail = top_tail_mean(val_continuation, args.tail_alpha)
            val_max = val_continuation.max()
            val_baseline_delta_loss = val_mean.new_tensor(0.0)
            val_baseline_delta_mean = val_mean.new_tensor(0.0)
            val_baseline_delta_tail = val_mean.new_tensor(0.0)
            val_baseline_delta_max = val_mean.new_tensor(0.0)
            val_baseline_delta_count = 0
            if args.selection_baseline_delta_weight > 0.0:
                base_val_buyer_regret, _ = continuation_regret_for_side(
                    baseline_model,
                    cfg,
                    validation_buyer_draws,
                    "buyer",
                    validation_coefficients,
                    validation_exit_thresholds,
                    args.chunk_size,
                )
                base_val_seller_regret, _ = continuation_regret_for_side(
                    baseline_model,
                    cfg,
                    validation_seller_draws,
                    "seller",
                    validation_coefficients,
                    validation_exit_thresholds,
                    args.chunk_size,
                )
                base_val_continuation = torch.cat([base_val_buyer_regret, base_val_seller_regret])
                (
                    val_baseline_delta_loss,
                    val_baseline_delta_mean,
                    val_baseline_delta_tail,
                    val_baseline_delta_max,
                    val_baseline_delta_count,
                ) = baseline_excess_stats(
                    val_continuation,
                    base_val_continuation,
                    args.tail_alpha,
                    args.baseline_delta_tail_weight,
                    args.baseline_delta_max_weight,
                    args.baseline_delta_margin,
                )
            val_hard_loss = val_mean.new_tensor(0.0)
            val_hard_mean = val_mean.new_tensor(0.0)
            val_hard_tail = val_mean.new_tensor(0.0)
            val_hard_max = val_mean.new_tensor(0.0)
            val_hard_count = 0
            val_hard_baseline_delta_loss = val_mean.new_tensor(0.0)
            if args.selection_hard_case_weight > 0.0 and validation_hard_case_draws:
                val_hard_samples = continuation_samples_from_bank(
                    model,
                    cfg,
                    validation_hard_case_draws,
                    validation_hard_case_coefficients,
                    validation_hard_case_exit_thresholds,
                    args.chunk_size,
                )
                if val_hard_samples is not None:
                    val_hard_mean = val_hard_samples.mean()
                    val_hard_tail = top_tail_mean(val_hard_samples, args.tail_alpha)
                    val_hard_max = val_hard_samples.max()
                    val_hard_count = int(val_hard_samples.numel())
                    val_hard_loss = val_hard_mean + args.hard_case_tail_weight * val_hard_tail + args.hard_case_max_weight * val_hard_max
                    if args.selection_baseline_delta_weight > 0.0:
                        base_val_hard_samples = continuation_samples_from_bank(
                            baseline_model,
                            cfg,
                            validation_hard_case_draws,
                            validation_hard_case_coefficients,
                            validation_hard_case_exit_thresholds,
                            args.chunk_size,
                        )
                        val_hard_baseline_delta_loss, _, _, _, _ = baseline_excess_stats(
                            val_hard_samples,
                            base_val_hard_samples,
                            args.tail_alpha,
                            args.baseline_delta_tail_weight,
                            args.baseline_delta_max_weight,
                            args.baseline_delta_margin,
                        )
            val_online_hard_loss = val_mean.new_tensor(0.0)
            val_online_hard_mean = val_mean.new_tensor(0.0)
            val_online_hard_tail = val_mean.new_tensor(0.0)
            val_online_hard_max = val_mean.new_tensor(0.0)
            val_online_hard_count = 0
            val_online_baseline_delta_loss = val_mean.new_tensor(0.0)
            if args.selection_online_hard_weight > 0.0 and online_hard_enabled:
                if args.fixed_validation_online_hard:
                    val_online_draws = fixed_validation_online_draws
                    val_online_coefficients = fixed_validation_online_coefficients
                    val_online_exit_thresholds = fixed_validation_online_exit_thresholds
                else:
                    assert validation_online_mining_coefficients is not None
                    assert validation_online_mining_exit_thresholds is not None
                    val_online_episodes = args.validation_online_hard_episodes or args.validation_episodes
                    val_online_draws, val_online_extra_coefficients, val_online_extra_exits, _val_online_manifest = mine_online_hard_cases(
                        model,
                        cfg,
                        validation_online_mining_coefficients,
                        validation_online_mining_exit_thresholds,
                        val_online_episodes,
                        args.online_hard_top_k,
                        seed + args.online_hard_seed_offset + 888_000 + step * 10_007,
                        device,
                        args.chunk_size,
                    )
                    val_online_coefficients, val_online_exit_thresholds = append_strategy_rows(
                        validation_coefficients,
                        validation_exit_thresholds,
                        val_online_extra_coefficients,
                        val_online_extra_exits,
                    )
                val_online_samples = continuation_samples_from_bank(
                    model,
                    cfg,
                    val_online_draws,
                    val_online_coefficients,
                    val_online_exit_thresholds,
                    args.chunk_size,
                )
                if val_online_samples is not None:
                    val_online_hard_mean = val_online_samples.mean()
                    val_online_hard_tail = top_tail_mean(val_online_samples, args.tail_alpha)
                    val_online_hard_max = val_online_samples.max()
                    val_online_hard_count = int(val_online_samples.numel())
                    val_online_hard_loss = (
                        val_online_hard_mean
                        + args.online_hard_tail_weight * val_online_hard_tail
                        + args.online_hard_max_weight * val_online_hard_max
                    )
                    if args.selection_baseline_delta_weight > 0.0:
                        base_val_online_samples = continuation_samples_from_bank(
                            baseline_model,
                            cfg,
                            val_online_draws,
                            val_online_coefficients,
                            val_online_exit_thresholds,
                            args.chunk_size,
                        )
                        val_online_baseline_delta_loss, _, _, _, _ = baseline_excess_stats(
                            val_online_samples,
                            base_val_online_samples,
                            args.tail_alpha,
                            args.baseline_delta_tail_weight,
                            args.baseline_delta_max_weight,
                            args.baseline_delta_margin,
                        )
            val_total_baseline_delta_loss = (
                val_baseline_delta_loss
                + val_hard_baseline_delta_loss
                + val_online_baseline_delta_loss
            )
            efficiency_floor = args.selection_efficiency_target * first_best_objective
            efficiency_penalty = val_sim["objective"].new_tensor(0.0)
            if args.selection_efficiency_target > 0.0:
                efficiency_penalty = torch.relu(val_sim["objective"].new_tensor(efficiency_floor) - val_sim["objective"])
            val_score_tensor = (
                val_mean
                + args.cont_tail_weight * val_tail
                + args.selection_max_weight * val_max
                + torch.relu(val_sim["mean_regret"] - cfg.regret_target)
                + args.p95_regret_weight * torch.relu(val_sim["p95_regret"] - args.p95_regret_target)
                + args.selection_efficiency_weight * efficiency_penalty
                + args.selection_hard_case_weight * val_hard_loss
                + args.selection_online_hard_weight * val_online_hard_loss
                + args.selection_baseline_delta_weight * val_total_baseline_delta_loss
                - args.selection_objective_weight * val_sim["objective"]
            )
        return {
            "validation_score": float(val_score_tensor.item()),
            "validation_objective": float(val_sim["objective"].item()),
            "validation_mean_regret": float(val_sim["mean_regret"].item()),
            "validation_p95_regret": float(val_sim["p95_regret"].item()),
            "validation_continuation_regret": float(val_mean.item()),
            "validation_continuation_tail": float(val_tail.item()),
            "validation_continuation_max": float(val_max.item()),
            "validation_hard_case_regret": float(val_hard_mean.item()),
            "validation_hard_case_tail": float(val_hard_tail.item()),
            "validation_hard_case_max": float(val_hard_max.item()),
            "validation_hard_case_count": val_hard_count,
            "validation_online_hard_regret": float(val_online_hard_mean.item()),
            "validation_online_hard_tail": float(val_online_hard_tail.item()),
            "validation_online_hard_max": float(val_online_hard_max.item()),
            "validation_online_hard_count": val_online_hard_count,
            "validation_baseline_delta_excess": float(val_baseline_delta_mean.item()),
            "validation_baseline_delta_tail": float(val_baseline_delta_tail.item()),
            "validation_baseline_delta_max": float(val_baseline_delta_max.item()),
            "validation_baseline_delta_count": val_baseline_delta_count,
            "validation_baseline_delta_loss": float(val_total_baseline_delta_loss.item()),
            "validation_efficiency_floor_gap": float(efficiency_penalty.item()),
        }

    history = []
    online_hard_history: list[dict[str, object]] = []
    online_hard_draws: dict[str, dict[str, torch.Tensor]] = {}
    online_hard_extra_coefficients: torch.Tensor | None = None
    online_hard_extra_exits: torch.Tensor | None = None
    online_hard_coefficients, online_hard_exit_thresholds = coefficients, strategy_exit_thresholds
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_step = 0
    baseline_selection_row: dict[str, float | int] | None = None
    if args.select_best and args.include_baseline_selection:
        baseline_selection_row = compute_selection_row(0)
        best_score = float(baseline_selection_row["validation_score"])
        print(
            f"baseline val={best_score:.4f} obj={baseline_selection_row['validation_objective']:.4f} "
            f"cont={baseline_selection_row['validation_continuation_regret']:.4f} "
            f"tail={baseline_selection_row['validation_continuation_tail']:.4f} "
            f"max={baseline_selection_row['validation_continuation_max']:.4f}",
            flush=True,
        )
    for step in range(1, args.steps + 1):
        if args.refresh_every > 0 and step > 1 and (step - 1) % args.refresh_every == 0:
            coefficients, strategy_exit_thresholds, report_count = make_strategies(seed + step * 1009)
        if online_hard_enabled and (step == 1 or (step - 1) % max(args.online_hard_refresh_every, 1) == 0):
            assert online_mining_coefficients is not None
            assert online_mining_exit_thresholds is not None
            mine_seed = seed + args.online_hard_seed_offset + step * 10_007
            online_hard_draws, online_hard_extra_coefficients, online_hard_extra_exits, online_manifest = mine_online_hard_cases(
                model,
                cfg,
                online_mining_coefficients,
                online_mining_exit_thresholds,
                args.online_hard_episodes,
                args.online_hard_top_k,
                mine_seed,
                device,
                args.chunk_size,
            )
            online_manifest["step"] = int(step)
            online_hard_history.append(online_manifest)
            online_hard_coefficients, online_hard_exit_thresholds = append_strategy_rows(
                coefficients,
                strategy_exit_thresholds,
                online_hard_extra_coefficients,
                online_hard_extra_exits,
            )
        model.train()
        optimizer.zero_grad()
        sim = simulate_dynamic(model, cfg, cfg.batch_size, train=True)
        buyer_draws = make_base_draws(cfg, args.cont_episodes, seed + step * 17 + 1, device)
        seller_draws = make_base_draws(cfg, args.cont_episodes, seed + step * 17 + 2, device)
        buyer_regret, _ = continuation_regret_for_side(
            model,
            cfg,
            buyer_draws,
            "buyer",
            coefficients,
            strategy_exit_thresholds,
            args.chunk_size,
        )
        seller_regret, _ = continuation_regret_for_side(
            model,
            cfg,
            seller_draws,
            "seller",
            coefficients,
            strategy_exit_thresholds,
            args.chunk_size,
        )
        continuation_samples = torch.cat([buyer_regret, seller_regret])
        continuation_mean = continuation_samples.mean()
        continuation_tail = top_tail_mean(continuation_samples, args.tail_alpha)
        continuation_max = continuation_samples.max()
        continuation_loss = continuation_mean + args.cont_tail_weight * continuation_tail + args.cont_max_weight * continuation_max
        baseline_delta_loss = continuation_loss.new_tensor(0.0)
        baseline_delta_mean = continuation_loss.new_tensor(0.0)
        baseline_delta_tail = continuation_loss.new_tensor(0.0)
        baseline_delta_max = continuation_loss.new_tensor(0.0)
        baseline_delta_count = 0
        if args.baseline_delta_weight > 0.0:
            guard_buyer_draws = buyer_draws
            guard_seller_draws = seller_draws
            guard_buyer_regret = buyer_regret
            guard_seller_regret = seller_regret
            if args.baseline_delta_episodes > 0:
                guard_buyer_draws = make_base_draws(
                    cfg,
                    args.baseline_delta_episodes,
                    seed + args.baseline_delta_seed_offset + step * 31 + 1,
                    device,
                )
                guard_seller_draws = make_base_draws(
                    cfg,
                    args.baseline_delta_episodes,
                    seed + args.baseline_delta_seed_offset + step * 31 + 2,
                    device,
                )
                guard_buyer_regret, _ = continuation_regret_for_side(
                    model,
                    cfg,
                    guard_buyer_draws,
                    "buyer",
                    coefficients,
                    strategy_exit_thresholds,
                    args.chunk_size,
                )
                guard_seller_regret, _ = continuation_regret_for_side(
                    model,
                    cfg,
                    guard_seller_draws,
                    "seller",
                    coefficients,
                    strategy_exit_thresholds,
                    args.chunk_size,
                )
            with torch.no_grad():
                base_buyer_regret, _ = continuation_regret_for_side(
                    baseline_model,
                    cfg,
                    guard_buyer_draws,
                    "buyer",
                    coefficients,
                    strategy_exit_thresholds,
                    args.chunk_size,
                )
                base_seller_regret, _ = continuation_regret_for_side(
                    baseline_model,
                    cfg,
                    guard_seller_draws,
                    "seller",
                    coefficients,
                    strategy_exit_thresholds,
                    args.chunk_size,
                )
                base_continuation_samples = torch.cat([base_buyer_regret, base_seller_regret])
            guard_continuation_samples = torch.cat([guard_buyer_regret, guard_seller_regret])
            (
                baseline_delta_loss,
                baseline_delta_mean,
                baseline_delta_tail,
                baseline_delta_max,
                baseline_delta_count,
            ) = baseline_excess_stats(
                guard_continuation_samples,
                base_continuation_samples,
                args.tail_alpha,
                args.baseline_delta_tail_weight,
                args.baseline_delta_max_weight,
                args.baseline_delta_margin,
            )
        hard_case_loss = continuation_loss.new_tensor(0.0)
        hard_case_mean = continuation_loss.new_tensor(0.0)
        hard_case_tail = continuation_loss.new_tensor(0.0)
        hard_case_max = continuation_loss.new_tensor(0.0)
        hard_case_count = 0
        hard_case_baseline_delta_loss = continuation_loss.new_tensor(0.0)
        if args.hard_case_weight > 0.0 and hard_case_draws:
            hard_case_samples = continuation_samples_from_bank(
                model,
                cfg,
                hard_case_draws,
                hard_case_coefficients,
                hard_case_exit_thresholds,
                args.chunk_size,
            )
            if hard_case_samples is not None:
                hard_case_mean = hard_case_samples.mean()
                hard_case_tail = top_tail_mean(hard_case_samples, args.tail_alpha)
                hard_case_max = hard_case_samples.max()
                hard_case_count = int(hard_case_samples.numel())
                hard_case_loss = hard_case_mean + args.hard_case_tail_weight * hard_case_tail + args.hard_case_max_weight * hard_case_max
                if args.baseline_delta_weight > 0.0:
                    with torch.no_grad():
                        base_hard_case_samples = continuation_samples_from_bank(
                            baseline_model,
                            cfg,
                            hard_case_draws,
                            hard_case_coefficients,
                            hard_case_exit_thresholds,
                            args.chunk_size,
                        )
                    hard_case_baseline_delta_loss, _, _, _, _ = baseline_excess_stats(
                        hard_case_samples,
                        base_hard_case_samples,
                        args.tail_alpha,
                        args.baseline_delta_tail_weight,
                        args.baseline_delta_max_weight,
                        args.baseline_delta_margin,
                    )
        online_hard_loss = continuation_loss.new_tensor(0.0)
        online_hard_mean = continuation_loss.new_tensor(0.0)
        online_hard_tail = continuation_loss.new_tensor(0.0)
        online_hard_max = continuation_loss.new_tensor(0.0)
        online_hard_count = 0
        online_baseline_delta_loss = continuation_loss.new_tensor(0.0)
        if args.online_hard_weight > 0.0 and online_hard_draws:
            online_hard_samples = continuation_samples_from_bank(
                model,
                cfg,
                online_hard_draws,
                online_hard_coefficients,
                online_hard_exit_thresholds,
                args.chunk_size,
            )
            if online_hard_samples is not None:
                online_hard_mean = online_hard_samples.mean()
                online_hard_tail = top_tail_mean(online_hard_samples, args.tail_alpha)
                online_hard_max = online_hard_samples.max()
                online_hard_count = int(online_hard_samples.numel())
                online_hard_loss = online_hard_mean + args.online_hard_tail_weight * online_hard_tail + args.online_hard_max_weight * online_hard_max
                if args.baseline_delta_weight > 0.0:
                    with torch.no_grad():
                        base_online_hard_samples = continuation_samples_from_bank(
                            baseline_model,
                            cfg,
                            online_hard_draws,
                            online_hard_coefficients,
                            online_hard_exit_thresholds,
                            args.chunk_size,
                        )
                    online_baseline_delta_loss, _, _, _, _ = baseline_excess_stats(
                        online_hard_samples,
                        base_online_hard_samples,
                        args.tail_alpha,
                        args.baseline_delta_tail_weight,
                        args.baseline_delta_max_weight,
                        args.baseline_delta_margin,
                    )
        total_baseline_delta_loss = baseline_delta_loss + hard_case_baseline_delta_loss + online_baseline_delta_loss
        anchor_loss = continuation_loss.new_tensor(0.0)
        if args.anchor_weight > 0.0:
            anchor_terms = []
            for name, parameter in model.named_parameters():
                anchor_terms.append((parameter - anchor_state[name]).pow(2).mean())
            anchor_loss = torch.stack(anchor_terms).mean() if anchor_terms else anchor_loss
        mean_regret_penalty = cfg.regret_weight * torch.relu(sim["mean_regret"] - cfg.regret_target)
        p95_regret_penalty = args.p95_regret_weight * torch.relu(sim["p95_regret"] - args.p95_regret_target)
        loss = (
            -sim["objective"]
            + mean_regret_penalty
            + p95_regret_penalty
            + args.cont_weight * continuation_loss
            + args.hard_case_weight * hard_case_loss
            + args.online_hard_weight * online_hard_loss
            + args.baseline_delta_weight * total_baseline_delta_loss
            + args.anchor_weight * anchor_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        row = {
            "step": step,
            "loss": float(loss.detach().item()),
            "objective": float(sim["objective"].detach().item()),
            "mean_regret": float(sim["mean_regret"].detach().item()),
            "p95_regret": float(sim["p95_regret"].detach().item()),
            "continuation_regret": float(continuation_mean.detach().item()),
            "continuation_tail": float(continuation_tail.detach().item()),
            "continuation_max": float(continuation_max.detach().item()),
            "hard_case_regret": float(hard_case_mean.detach().item()),
            "hard_case_tail": float(hard_case_tail.detach().item()),
            "hard_case_max": float(hard_case_max.detach().item()),
            "hard_case_count": hard_case_count,
            "online_hard_regret": float(online_hard_mean.detach().item()),
            "online_hard_tail": float(online_hard_tail.detach().item()),
            "online_hard_max": float(online_hard_max.detach().item()),
            "online_hard_count": online_hard_count,
            "online_hard_strategy_count": int(online_hard_coefficients.shape[0]) if online_hard_draws else 0,
            "baseline_delta_excess": float(baseline_delta_mean.detach().item()),
            "baseline_delta_tail": float(baseline_delta_tail.detach().item()),
            "baseline_delta_max": float(baseline_delta_max.detach().item()),
            "baseline_delta_count": baseline_delta_count,
            "baseline_delta_loss": float(total_baseline_delta_loss.detach().item()),
            "anchor_loss": float(anchor_loss.detach().item()),
            "mean_regret_penalty": float(mean_regret_penalty.detach().item()),
            "p95_regret_penalty": float(p95_regret_penalty.detach().item()),
            "strategy_count": int(coefficients.shape[0]),
        }
        if args.select_best and (step == 1 or step % max(args.validation_every, 1) == 0 or step == args.steps):
            row.update(compute_selection_row(step))
            val_score = float(row["validation_score"])
            if val_score < best_score - args.baseline_selection_margin:
                best_score = val_score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                f"step={step:03d} obj={row['objective']:.4f} regret={row['mean_regret']:.4f} "
                f"p95={row['p95_regret']:.4f} cont={row['continuation_regret']:.4f} "
                f"tail={row['continuation_tail']:.4f} max={row['continuation_max']:.4f}"
                + (f" hard={row['hard_case_regret']:.4f}/{row['hard_case_max']:.4f}" if row["hard_case_count"] else "")
                + (f" online={row['online_hard_regret']:.4f}/{row['online_hard_max']:.4f}" if row["online_hard_count"] else "")
                + (f" delta={row['baseline_delta_excess']:.4f}/{row['baseline_delta_max']:.4f}" if row["baseline_delta_count"] else "")
                + (f" val={row['validation_score']:.4f}" if "validation_score" in row else ""),
                flush=True,
            )

    if args.select_best:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out_dir / "finetune_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "online_hard_history.json").write_text(json.dumps(online_hard_history, indent=2), encoding="utf-8")
    if baseline_selection_row is not None:
        (out_dir / "baseline_selection.json").write_text(json.dumps(baseline_selection_row, indent=2), encoding="utf-8")
    metrics = evaluate_and_save(model, cfg, out_dir)
    manifest = {
        "base_run": str(run_dir),
        "report_strategy_count": int(report_count),
        "continuation_strategy_count": int(coefficients.shape[0]),
        "exit_thresholds": [float(x) for x in exit_thresholds.tolist()],
        "continuation_objective": {
            "cont_weight": args.cont_weight,
            "cont_tail_weight": args.cont_tail_weight,
            "cont_max_weight": args.cont_max_weight,
            "hard_case_csv": args.hard_case_csv,
            "hard_case_top_k": args.hard_case_top_k,
            "hard_case_weight": args.hard_case_weight,
            "hard_case_tail_weight": args.hard_case_tail_weight,
            "hard_case_max_weight": args.hard_case_max_weight,
            "hard_case_bank": hard_case_manifest,
            "validation_hard_case_csv": args.validation_hard_case_csv,
            "validation_hard_case_bank": validation_hard_case_manifest,
            "selection_hard_case_weight": args.selection_hard_case_weight,
            "anchor_weight": args.anchor_weight,
            "online_hard_weight": args.online_hard_weight,
            "online_hard_tail_weight": args.online_hard_tail_weight,
            "online_hard_max_weight": args.online_hard_max_weight,
            "online_hard_refresh_every": args.online_hard_refresh_every,
            "online_hard_episodes": args.online_hard_episodes,
            "online_hard_top_k": args.online_hard_top_k,
            "online_hard_history_draws": args.online_hard_history_draws or args.history_draws,
            "online_hard_grid": args.online_hard_grid or args.grid,
            "online_hard_radius": args.online_hard_radius,
            "online_hard_seed_offset": args.online_hard_seed_offset,
            "online_mining_report_count": online_mining_report_count,
            "selection_online_hard_weight": args.selection_online_hard_weight,
            "validation_online_hard_episodes": args.validation_online_hard_episodes or args.validation_episodes,
            "fixed_validation_online_hard": args.fixed_validation_online_hard,
            "fixed_validation_online_hard_pool": fixed_validation_online_manifest,
            "online_hard_refresh_count": len(online_hard_history),
            "tail_alpha": args.tail_alpha,
            "p95_regret_weight": args.p95_regret_weight,
            "p95_regret_target": args.p95_regret_target,
            "refresh_every": args.refresh_every,
            "select_best": args.select_best,
            "include_baseline_selection": args.include_baseline_selection,
            "baseline_selection_margin": args.baseline_selection_margin,
            "baseline_delta_weight": args.baseline_delta_weight,
            "baseline_delta_tail_weight": args.baseline_delta_tail_weight,
            "baseline_delta_max_weight": args.baseline_delta_max_weight,
            "baseline_delta_margin": args.baseline_delta_margin,
            "baseline_delta_episodes": args.baseline_delta_episodes,
            "baseline_delta_seed_offset": args.baseline_delta_seed_offset,
            "selection_baseline_delta_weight": args.selection_baseline_delta_weight,
            "baseline_selection": baseline_selection_row,
            "best_step": best_step,
            "best_validation_score": best_score,
            "validation_every": args.validation_every,
            "validation_episodes": args.validation_episodes,
            "validation_objective_episodes": args.validation_objective_episodes,
            "selection_objective_weight": args.selection_objective_weight,
            "selection_max_weight": args.selection_max_weight,
            "selection_efficiency_target": args.selection_efficiency_target,
            "selection_efficiency_weight": args.selection_efficiency_weight,
            "first_best_objective": first_best_objective,
        },
        "metrics": metrics,
    }
    (out_dir / "finetune_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
