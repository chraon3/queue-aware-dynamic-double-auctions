from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch import nn

from analyze_queue_abandonment import load_config, load_neural, make_draws, simulate_with_decomposition
from dynamic_double_auction import DynamicConfig, set_seed, simulate_dynamic, terminal_pressure_cutoff


class TriggerComponentMcAfeeMechanism(nn.Module):
    """Queue-aware McAfee variants used only for trigger component ablations."""

    supports_state = True

    def __init__(self, cfg: DynamicConfig, variant: str, random_share: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.random_share = float(max(0.0, min(1.0, random_share)))

    def trigger(self, public_state: torch.Tensor) -> torch.Tensor:
        mean_age_pressure = 0.5 * (public_state[:, 3] + public_state[:, 4])
        imbalance = torch.abs(public_state[:, 2])
        terminal = public_state[:, 8] >= terminal_pressure_cutoff(self.cfg)
        age = mean_age_pressure >= self.cfg.queue_age_trigger
        imbalanced = imbalance >= self.cfg.queue_imbalance_trigger
        if self.variant == "full":
            return age | imbalanced | terminal
        if self.variant == "age_only":
            return age
        if self.variant == "imbalance_only":
            return imbalanced
        if self.variant == "terminal_only":
            return terminal
        if self.variant == "age_imbalance":
            return age | imbalanced
        if self.variant == "random_same_share":
            weights = torch.tensor(
                [12.9898, 78.233, 37.719, 19.913, 53.127, 91.441, 27.317, 7.733, 61.771, 43.113, 5.971, 31.337],
                device=public_state.device,
                dtype=public_state.dtype,
            )
            hashed = torch.frac(torch.sin((public_state * weights).sum(dim=1) + 0.173) * 43758.5453)
            hashed = torch.where(hashed < 0.0, hashed + 1.0, hashed)
            return hashed <= self.random_share
        if self.variant == "always_efficient":
            return torch.ones(public_state.shape[0], device=public_state.device, dtype=torch.bool)
        if self.variant == "always_mcafee":
            return torch.zeros(public_state.shape[0], device=public_state.device, dtype=torch.bool)
        raise ValueError(f"Unknown trigger variant: {self.variant}")

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

        trigger = self.trigger(public_state)
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
            if efficient_k <= 0:
                continue
            mcafee_k = max(efficient_k - 1, 0)
            if efficient_k < n_buyers and efficient_k < n_sellers:
                price = 0.5 * (sorted_buyers[efficient_k] + sorted_sellers[efficient_k])
                if sorted_sellers[efficient_k - 1] <= price <= sorted_buyers[efficient_k - 1]:
                    mcafee_k = efficient_k
            trade_k = efficient_k if bool(trigger[row].item()) else mcafee_k
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
        return {
            "match": match,
            "buyer_alloc": buyer_alloc,
            "seller_alloc": seller_alloc,
            "buyer_payments": buyer_payments,
            "seller_transfers": seller_transfers,
            "buyer_unit_payment": torch.zeros_like(match),
            "seller_unit_transfer": torch.zeros_like(match),
        }


def discover_runs(patterns: list[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        runs.extend(sorted(Path().glob(pattern)))
    return [path for path in runs if (path / "config.json").exists() and (path / "model.pt").exists()]


def trigger_variants() -> list[dict[str, str]]:
    return [
        {"setting": "Full trigger", "variant": "full"},
        {"setting": "Age only", "variant": "age_only"},
        {"setting": "Imbalance only", "variant": "imbalance_only"},
        {"setting": "Terminal only", "variant": "terminal_only"},
        {"setting": "Age + imbalance", "variant": "age_imbalance"},
        {"setting": "Random same share", "variant": "random_same_share"},
        {"setting": "Always efficient", "variant": "always_efficient"},
        {"setting": "Always McAfee", "variant": "always_mcafee"},
    ]


def regret_metrics(cfg: DynamicConfig, variant: str, random_share: float, episodes: int, seed: int) -> dict[str, float]:
    audit_cfg = replace(cfg, eval_episodes=episodes, seed=seed, device="cpu")
    set_seed(audit_cfg.seed)
    model = TriggerComponentMcAfeeMechanism(audit_cfg, variant, random_share).to(torch.device(audit_cfg.device)).eval()
    with torch.no_grad():
        sim = simulate_dynamic(model, audit_cfg, audit_cfg.eval_episodes, train=True)
    return {
        "mean_regret": float(sim["mean_regret"].item()),
        "p95_regret": float(sim["p95_regret"].item()),
        "max_regret": float(sim["max_regret"].item()),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    order = {row["setting"]: idx for idx, row in enumerate(trigger_variants())}
    rows: list[Dict[str, float | str | int]] = []
    for setting, group in raw.groupby("setting", sort=False):
        row: Dict[str, float | str | int] = {
            "setting": setting,
            "variant": str(group["variant"].iloc[0]),
            "runs": int(group["seed"].nunique()),
            "setting_order": order[str(setting)],
        }
        for metric in [
            "queue_objective",
            "neural_objective",
            "objective_gap",
            "abandonment_gap",
            "volume_gap",
            "random_share",
            "mean_regret",
            "p95_regret",
            "max_regret",
        ]:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["objective_wins"] = int((group["objective_gap"] > 0.0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("setting_order").drop(columns=["setting_order"]).reset_index(drop=True)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    def fmt(value: float) -> str:
        if pd.isna(value):
            return "--"
        return f"{value:.3f}"

    rows = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Rule & Obj. gap & Wins & Aband. gap & Vol. gap & Mean reg. & P95 reg. \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        rows.append(
            f"{row['setting']} & "
            f"{fmt(row['objective_gap'])} & "
            f"{int(row['objective_wins'])}/10 & "
            f"{fmt(row['abandonment_gap'])} & "
            f"{fmt(row['volume_gap'])} & "
            f"{fmt(row['mean_regret'])} & "
            f"{fmt(row['p95_regret'])} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/dynamic_patience_value_seed*"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--regret-episodes", type=int, default=240)
    parser.add_argument("--draw-seed-offset", type=int, default=970000)
    parser.add_argument("--regret-seed-offset", type=int, default=980000)
    parser.add_argument("--out-dir", default="experiments/queue_trigger_component_ablation")
    parser.add_argument("--paper-table", default="paper/tables/queue_trigger_component_ablation.tex")
    args = parser.parse_args()

    run_dirs = discover_runs(args.runs)
    if not run_dirs:
        raise FileNotFoundError("No dynamic run directories found.")

    rows: list[dict[str, float | int | str]] = []
    variants = trigger_variants()
    for run_idx, run_dir in enumerate(run_dirs):
        base_cfg = load_config(run_dir / "config.json")
        cfg = replace(base_cfg, eval_episodes=args.episodes, device="cpu")
        device = torch.device(cfg.device)
        draws = make_draws(cfg, args.episodes, args.draw_seed_offset + cfg.seed, device)
        neural = load_neural(run_dir, cfg)
        with torch.no_grad():
            neural_row = simulate_with_decomposition(neural, cfg, draws, "Neural dynamic mechanism", cfg.seed)

        full_model = TriggerComponentMcAfeeMechanism(cfg, "full").to(device).eval()
        with torch.no_grad():
            full_row = simulate_with_decomposition(full_model, cfg, draws, "Full trigger", cfg.seed)
        random_share = float(full_row["high_pressure_state_share"])

        cached_rows = {"full": full_row}
        for variant_idx, setting in enumerate(variants):
            variant = setting["variant"]
            if variant in cached_rows:
                queue_row = cached_rows[variant]
            else:
                model = TriggerComponentMcAfeeMechanism(cfg, variant, random_share).to(device).eval()
                with torch.no_grad():
                    queue_row = simulate_with_decomposition(model, cfg, draws, str(setting["setting"]), cfg.seed)
            regrets = regret_metrics(
                base_cfg,
                variant,
                random_share,
                args.regret_episodes,
                args.regret_seed_offset + cfg.seed + 1009 * variant_idx + 7919 * run_idx,
            )
            row = {
                "run": run_dir.name,
                "seed": cfg.seed,
                "setting": str(setting["setting"]),
                "variant": variant,
                "random_share": random_share if variant == "random_same_share" else float("nan"),
                "neural_objective": float(neural_row["objective"]),
                "queue_objective": float(queue_row["objective"]),
                "objective_gap": float(queue_row["objective"]) - float(neural_row["objective"]),
                "neural_abandonment": float(neural_row["abandonment"]),
                "queue_abandonment": float(queue_row["abandonment"]),
                "abandonment_gap": float(queue_row["abandonment"]) - float(neural_row["abandonment"]),
                "volume_gap": float(queue_row["volume"]) - float(neural_row["volume"]),
                "unmatched_gap": float(queue_row["unmatched"]) - float(neural_row["unmatched"]),
                **regrets,
            }
            rows.append(row)
            print(
                f"ablation seed={cfg.seed} setting={setting['setting']} "
                f"obj_gap={row['objective_gap']:.3f} aband_gap={row['abandonment_gap']:.3f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(out_dir / "queue_trigger_component_ablation_by_seed.csv", index=False)
    summary.to_csv(out_dir / "queue_trigger_component_ablation_summary.csv", index=False)
    write_latex(summary, Path(args.paper_table))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
