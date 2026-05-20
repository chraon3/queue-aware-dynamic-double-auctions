from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch import nn

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


def load_config(path: Path) -> DynamicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = DynamicConfig.__dataclass_fields__.keys()
    return DynamicConfig(**{key: value for key, value in data.items() if key in valid})


def load_neural(run_dir: Path, cfg: DynamicConfig) -> nn.Module:
    device = torch.device(cfg.device)
    model = build_dynamic_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return model


def make_draws(cfg: DynamicConfig, episodes: int, seed: int, device: torch.device) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return {
        "buyer_arrival": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "seller_arrival": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "buyer_value": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "seller_cost": torch.rand(cfg.horizon, episodes, generator=generator, device=device),
        "buyer_abandon": torch.rand(cfg.horizon, episodes, cfg.max_buyers, generator=generator, device=device),
        "seller_abandon": torch.rand(cfg.horizon, episodes, cfg.max_sellers, generator=generator, device=device),
    }


def add_arrivals_from_draws(
    values: torch.Tensor,
    active: torch.Tensor,
    ages: torch.Tensor,
    arrival_draw: torch.Tensor,
    value_draw: torch.Tensor,
    arrival_prob: float,
    fill_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _ = values.shape
    for idx in range(batch):
        if arrival_draw[idx].item() <= arrival_prob:
            open_slots = torch.nonzero(active[idx] < 0.5, as_tuple=False).flatten()
            if open_slots.numel() > 0:
                slot = int(open_slots[0].item())
                values[idx, slot] = value_draw[idx]
                active[idx, slot] = 1.0
                ages[idx, slot] = 0.0
    values = torch.where(active > 0.5, values, torch.full_like(values, fill_value))
    ages = torch.where(active > 0.5, ages, torch.zeros_like(ages))
    return values, active, ages


def safe_ratio(num: torch.Tensor, den: torch.Tensor) -> float:
    if float(den.item()) <= 1.0e-12:
        return float("nan")
    return float((num / den).item())


def simulate_with_decomposition(
    model: nn.Module,
    cfg: DynamicConfig,
    draws: Dict[str, torch.Tensor],
    policy: str,
    seed: int,
) -> Dict[str, float | str | int]:
    device = torch.device(cfg.device)
    episodes = draws["buyer_arrival"].shape[1]
    buyer_values = torch.zeros(episodes, cfg.max_buyers, device=device)
    seller_costs = torch.ones(episodes, cfg.max_sellers, device=device)
    buyer_active = torch.zeros_like(buyer_values)
    seller_active = torch.zeros_like(seller_costs)
    buyer_ages = torch.zeros_like(buyer_values)
    seller_ages = torch.zeros_like(seller_costs)

    discount_rewards = []
    surplus_sum = torch.tensor(0.0, device=device)
    wait_sum = torch.tensor(0.0, device=device)
    volume_sum = torch.tensor(0.0, device=device)
    queue_sum = torch.tensor(0.0, device=device)
    unmatched_sum = torch.tensor(0.0, device=device)
    state_count = torch.tensor(float(cfg.horizon * episodes), device=device)

    buyer_abandon_sum = torch.tensor(0.0, device=device)
    seller_abandon_sum = torch.tensor(0.0, device=device)
    forced_sum = torch.tensor(0.0, device=device)
    stochastic_sum = torch.tensor(0.0, device=device)
    high_pressure_abandon_sum = torch.tensor(0.0, device=device)
    low_pressure_abandon_sum = torch.tensor(0.0, device=device)
    at_risk_sum = torch.tensor(0.0, device=device)
    high_state_sum = torch.tensor(0.0, device=device)
    age_at_abandon_sum = torch.tensor(0.0, device=device)
    abandoned_value_sum = torch.tensor(0.0, device=device)
    abandoned_count = torch.tensor(0.0, device=device)
    abandoned_opportunity_sum = torch.tensor(0.0, device=device)

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

        pair_surplus = buyer_values[:, :, None] - seller_costs[:, None, :]
        surplus = (match * pair_surplus).sum(dim=(1, 2))
        unmatched_buyers = buyer_active * (1.0 - buyer_alloc)
        unmatched_sellers = seller_active * (1.0 - seller_alloc)
        wait_penalty = cfg.wait_cost * (unmatched_buyers.sum(dim=1) + unmatched_sellers.sum(dim=1))
        reward = surplus - wait_penalty
        discount_rewards.append((cfg.discount**t) * reward)

        surplus_sum = surplus_sum + surplus.sum()
        wait_sum = wait_sum + wait_penalty.sum()
        volume_sum = volume_sum + match.sum()
        queue_sum = queue_sum + buyer_active.sum() + seller_active.sum()
        unmatched_sum = unmatched_sum + unmatched_buyers.sum() + unmatched_sellers.sum()

        depart_buyers = buyer_alloc.detach() > 0.5
        depart_sellers = seller_alloc.detach() > 0.5
        buyer_ages_after_wait = buyer_ages + unmatched_buyers.detach()
        seller_ages_after_wait = seller_ages + unmatched_sellers.detach()
        buyer_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * buyer_ages_after_wait / max(cfg.max_patience, 1), 0.0, 1.0)
        seller_prob = torch.clamp(cfg.abandon_base + cfg.abandon_slope * seller_ages_after_wait / max(cfg.max_patience, 1), 0.0, 1.0)

        buyer_forced = buyer_ages_after_wait >= cfg.max_patience
        seller_forced = seller_ages_after_wait >= cfg.max_patience
        buyer_stochastic = (draws["buyer_abandon"][t] < buyer_prob) & (~buyer_forced)
        seller_stochastic = (draws["seller_abandon"][t] < seller_prob) & (~seller_forced)
        buyer_at_risk = (buyer_active > 0.5) & (~depart_buyers)
        seller_at_risk = (seller_active > 0.5) & (~depart_sellers)
        buyer_abandon = (buyer_forced | buyer_stochastic) & buyer_at_risk
        seller_abandon = (seller_forced | seller_stochastic) & seller_at_risk

        abandonment_by_episode = buyer_abandon.float().sum(dim=1) + seller_abandon.float().sum(dim=1)
        mean_age_pressure = 0.5 * (public_state[:, 3] + public_state[:, 4])
        imbalance = public_state[:, 2].abs()
        high_pressure = queue_pressure_trigger(mean_age_pressure, imbalance, public_state[:, 8], cfg)
        high_state_sum = high_state_sum + high_pressure.float().sum()
        high_pressure_abandon_sum = high_pressure_abandon_sum + abandonment_by_episode[high_pressure].sum()
        low_pressure_abandon_sum = low_pressure_abandon_sum + abandonment_by_episode[~high_pressure].sum()

        buyer_abandon_count = buyer_abandon.float().sum()
        seller_abandon_count = seller_abandon.float().sum()
        buyer_abandon_sum = buyer_abandon_sum + buyer_abandon_count
        seller_abandon_sum = seller_abandon_sum + seller_abandon_count
        forced_sum = forced_sum + (buyer_forced & buyer_at_risk).float().sum() + (seller_forced & seller_at_risk).float().sum()
        stochastic_sum = stochastic_sum + (buyer_stochastic & buyer_at_risk).float().sum() + (seller_stochastic & seller_at_risk).float().sum()
        at_risk_sum = at_risk_sum + buyer_at_risk.float().sum() + seller_at_risk.float().sum()

        abandoned_count_t = buyer_abandon_count + seller_abandon_count
        if abandoned_count_t.item() > 0:
            age_at_abandon_sum = age_at_abandon_sum + buyer_ages_after_wait[buyer_abandon].sum() + seller_ages_after_wait[seller_abandon].sum()
            abandoned_value_sum = abandoned_value_sum + buyer_values[buyer_abandon].sum() + (1.0 - seller_costs[seller_abandon]).sum()
            abandoned_count = abandoned_count + abandoned_count_t

            min_seller = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs)).min(dim=1).values
            max_buyer = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values)).max(dim=1).values
            buyer_opportunity = torch.relu(buyer_values - min_seller[:, None])
            seller_opportunity = torch.relu(max_buyer[:, None] - seller_costs)
            abandoned_opportunity_sum = abandoned_opportunity_sum + buyer_opportunity[buyer_abandon].sum() + seller_opportunity[seller_abandon].sum()

        buyer_ages = buyer_ages_after_wait
        seller_ages = seller_ages_after_wait
        buyer_active = torch.where(depart_buyers | buyer_abandon, torch.zeros_like(buyer_active), buyer_active)
        seller_active = torch.where(depart_sellers | seller_abandon, torch.zeros_like(seller_active), seller_active)
        buyer_values = torch.where(buyer_active > 0.5, buyer_values, torch.zeros_like(buyer_values))
        seller_costs = torch.where(seller_active > 0.5, seller_costs, torch.ones_like(seller_costs))
        buyer_ages = torch.where(buyer_active > 0.5, buyer_ages, torch.zeros_like(buyer_ages))
        seller_ages = torch.where(seller_active > 0.5, seller_ages, torch.zeros_like(seller_ages))

    total_reward = torch.stack(discount_rewards, dim=0).sum(dim=0)
    total_abandon = buyer_abandon_sum + seller_abandon_sum
    return {
        "seed": seed,
        "policy": policy,
        "objective": float(total_reward.mean().item()),
        "surplus": safe_ratio(surplus_sum, state_count),
        "wait_cost": safe_ratio(wait_sum, state_count),
        "volume": safe_ratio(volume_sum, state_count),
        "queue_length": safe_ratio(queue_sum, state_count),
        "unmatched": safe_ratio(unmatched_sum, state_count),
        "abandonment": safe_ratio(total_abandon, state_count),
        "buyer_abandonment": safe_ratio(buyer_abandon_sum, state_count),
        "seller_abandonment": safe_ratio(seller_abandon_sum, state_count),
        "forced_abandonment": safe_ratio(forced_sum, state_count),
        "stochastic_abandonment": safe_ratio(stochastic_sum, state_count),
        "forced_share": safe_ratio(forced_sum, total_abandon),
        "buyer_share": safe_ratio(buyer_abandon_sum, total_abandon),
        "high_pressure_state_share": safe_ratio(high_state_sum, state_count),
        "high_pressure_abandon_share": safe_ratio(high_pressure_abandon_sum, total_abandon),
        "low_pressure_abandon_share": safe_ratio(low_pressure_abandon_sum, total_abandon),
        "at_risk_abandon_rate": safe_ratio(total_abandon, at_risk_sum),
        "mean_age_at_abandon": safe_ratio(age_at_abandon_sum, abandoned_count),
        "mean_abandoner_quality": safe_ratio(abandoned_value_sum, abandoned_count),
        "mean_abandoner_opportunity": safe_ratio(abandoned_opportunity_sum, abandoned_count),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "objective",
        "volume",
        "queue_length",
        "unmatched",
        "abandonment",
        "buyer_abandonment",
        "seller_abandonment",
        "forced_share",
        "high_pressure_abandon_share",
        "at_risk_abandon_rate",
        "mean_age_at_abandon",
        "mean_abandoner_opportunity",
    ]
    rows = []
    for policy, group in raw.groupby("policy", sort=False):
        row: Dict[str, float | str] = {"policy": policy}
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    if raw["policy"].nunique() == 2:
        pivot = raw.pivot(index="seed", columns="policy")
        policies = list(raw["policy"].drop_duplicates())
        delta: Dict[str, float | str] = {"policy": f"{policies[1]} minus {policies[0]}"}
        for metric in metrics:
            diff = pivot[metric][policies[1]] - pivot[metric][policies[0]]
            delta[metric] = float(diff.mean())
            delta[f"{metric}_std"] = float(diff.std(ddof=1))
        rows.append(delta)
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    def fmt(value: float) -> str:
        if pd.isna(value):
            return "--"
        return f"{value:.3f}"

    label_map = {
        "Neural dynamic mechanism": "Neural",
        "Payment-audited queue-aware McAfee": "Audited queue McAfee",
        "Payment-audited queue-aware McAfee minus Neural dynamic mechanism": "Difference",
    }
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Policy & Obj. & Aband. & Forced share & High-pressure share & At-risk rate & Exit opp. \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{label_map.get(str(row['policy']), str(row['policy']))} & "
            f"{fmt(row['objective'])} & "
            f"{fmt(row['abandonment'])} & "
            f"{fmt(row['forced_share'])} & "
            f"{fmt(row['high_pressure_abandon_share'])} & "
            f"{fmt(row['at_risk_abandon_rate'])} & "
            f"{fmt(row['mean_abandoner_opportunity'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--draw-seed-offset", type=int, default=710000)
    parser.add_argument("--out-dir", default="experiments/queue_abandonment_decomposition")
    parser.add_argument("--paper-table", default="paper/tables/queue_abandonment_decomposition.tex")
    args = parser.parse_args()

    run_dirs = []
    for pattern in args.runs:
        run_dirs.extend(sorted(Path().glob(pattern)))
    run_dirs = [path for path in run_dirs if (path / "config.json").exists() and (path / "model.pt").exists()]
    if not run_dirs:
        raise FileNotFoundError("No dynamic run directories found.")

    rows = []
    for run_dir in run_dirs:
        cfg = load_config(run_dir / "config.json")
        cfg = replace(cfg, eval_episodes=args.episodes, device="cpu")
        set_seed(cfg.seed)
        device = torch.device(cfg.device)
        draws = make_draws(cfg, args.episodes, args.draw_seed_offset + cfg.seed, device)
        neural = load_neural(run_dir, cfg)
        queue = QueueAwareMcAfeeMechanism(cfg).to(device).eval()
        with torch.no_grad():
            rows.append(simulate_with_decomposition(neural, cfg, draws, "Neural dynamic mechanism", cfg.seed))
            rows.append(simulate_with_decomposition(queue, cfg, draws, "Payment-audited queue-aware McAfee", cfg.seed))
        print(f"decomposed seed={cfg.seed}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows).sort_values(["seed", "policy"]).reset_index(drop=True)
    summary = summarize(raw)
    raw.to_csv(out_dir / "queue_abandonment_by_seed.csv", index=False)
    summary.to_csv(out_dir / "queue_abandonment_summary.csv", index=False)

    table_path = Path(args.paper_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_latex(summary, table_path)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
