from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict

import numpy as np
import pandas as pd


Agent = tuple[int, int]
SideState = tuple[Agent, ...]
FullState = tuple[SideState, SideState]


@dataclass(frozen=True)
class DPConfig:
    values: tuple[float, ...] = (0.2, 0.5, 0.8)
    max_buyers: int = 2
    max_sellers: int = 2
    horizon: int = 5
    arrival_prob_buyer: float = 0.72
    arrival_prob_seller: float = 0.72
    wait_cost: float = 0.015
    discount: float = 0.97
    max_patience: int = 5
    abandon_base: float = 0.015
    abandon_slope: float = 0.12
    imbalance_trigger: float = 0.25
    age_trigger: float = 0.40
    terminal_window: int = 2


def canonical_buyers(buyers: SideState) -> SideState:
    return tuple(sorted(buyers, key=lambda x: (-x[0], -x[1])))


def canonical_sellers(sellers: SideState) -> SideState:
    return tuple(sorted(sellers, key=lambda x: (x[0], -x[1])))


class RelaxationDP:
    def __init__(self, cfg: DPConfig) -> None:
        self.cfg = cfg

    def value(self, idx: int) -> float:
        return self.cfg.values[idx]

    def canonical(self, state: FullState) -> FullState:
        return canonical_buyers(state[0]), canonical_sellers(state[1])

    @lru_cache(maxsize=None)
    def arrival_outcomes(self, side: SideState, is_buyer: bool) -> tuple[tuple[SideState, float], ...]:
        cap = self.cfg.max_buyers if is_buyer else self.cfg.max_sellers
        canonical = canonical_buyers if is_buyer else canonical_sellers
        if len(side) >= cap:
            return ((side, 1.0),)
        outcomes: DefaultDict[SideState, float] = defaultdict(float)
        p = self.cfg.arrival_prob_buyer if is_buyer else self.cfg.arrival_prob_seller
        outcomes[side] += 1.0 - p
        arrival_weight = p / len(self.cfg.values)
        for idx in range(len(self.cfg.values)):
            outcomes[canonical(side + ((idx, 0),))] += arrival_weight
        return tuple(outcomes.items())

    def sorted_values(self, buyers: SideState, sellers: SideState) -> tuple[list[float], list[float]]:
        return [self.value(idx) for idx, _ in canonical_buyers(buyers)], [
            self.value(idx) for idx, _ in canonical_sellers(sellers)
        ]

    def efficient_count(self, buyers: SideState, sellers: SideState) -> int:
        b_vals, s_vals = self.sorted_values(buyers, sellers)
        return sum(b >= s for b, s in zip(b_vals, s_vals))

    def mcafee_count(self, buyers: SideState, sellers: SideState) -> int:
        b_vals, s_vals = self.sorted_values(buyers, sellers)
        efficient_k = self.efficient_count(buyers, sellers)
        if efficient_k == 0:
            return 0
        trade_k = efficient_k - 1
        if efficient_k < len(b_vals) and efficient_k < len(s_vals):
            price = 0.5 * (b_vals[efficient_k] + s_vals[efficient_k])
            if s_vals[efficient_k - 1] <= price <= b_vals[efficient_k - 1]:
                trade_k = efficient_k
        return max(trade_k, 0)

    def is_congested(self, buyers: SideState, sellers: SideState, t: int) -> bool:
        ages = [age for _, age in buyers + sellers]
        mean_age = float(np.mean(ages)) if ages else 0.0
        total_capacity = max(self.cfg.max_buyers + self.cfg.max_sellers, 1)
        imbalance = abs(len(buyers) - len(sellers)) / total_capacity
        late_market = float(t) / max(self.cfg.horizon - 1, 1) >= 1.0 - self.cfg.terminal_window / max(
            self.cfg.horizon - 1,
            1,
        )
        return (
            mean_age >= self.cfg.age_trigger * self.cfg.max_patience
            or imbalance >= self.cfg.imbalance_trigger
            or late_market
        )

    def queue_aware_count(self, buyers: SideState, sellers: SideState, t: int) -> int:
        efficient_k = self.efficient_count(buyers, sellers)
        mcafee_k = self.mcafee_count(buyers, sellers)
        if efficient_k <= mcafee_k:
            return mcafee_k
        return efficient_k if self.is_congested(buyers, sellers, t) else mcafee_k

    def trade_count(self, buyers: SideState, sellers: SideState, t: int, policy: str) -> int:
        if policy == "first_best":
            return self.efficient_count(buyers, sellers)
        if policy == "mcafee":
            return self.mcafee_count(buyers, sellers)
        if policy == "queue_aware":
            return self.queue_aware_count(buyers, sellers, t)
        raise ValueError(policy)

    def remove_trades(self, buyers: SideState, sellers: SideState, k: int) -> FullState:
        b_sorted = canonical_buyers(buyers)
        s_sorted = canonical_sellers(sellers)
        return canonical_buyers(b_sorted[k:]), canonical_sellers(s_sorted[k:])

    def surplus_for_count(self, buyers: SideState, sellers: SideState, k: int) -> float:
        b_vals, s_vals = self.sorted_values(buyers, sellers)
        return float(sum(b_vals[pos] - s_vals[pos] for pos in range(k)))

    def survival_probability(self, age: int, periods: int) -> float:
        prob = 1.0
        for step in range(1, periods + 1):
            new_age = age + step
            if new_age >= self.cfg.max_patience:
                return 0.0
            hazard = min(
                self.cfg.abandon_base + self.cfg.abandon_slope * new_age / max(self.cfg.max_patience, 1),
                1.0,
            )
            prob *= 1.0 - hazard
        return prob

    def discounted_opportunity_factor(self, age: int, arrival_prob: float, remaining_periods: int) -> float:
        total = 0.0
        no_prior_arrival = 1.0
        for step in range(1, remaining_periods + 1):
            survive = self.survival_probability(age, step)
            first_arrival = no_prior_arrival * arrival_prob
            total += (self.cfg.discount**step) * survive * first_arrival
            no_prior_arrival *= 1.0 - arrival_prob
        return total

    def discounted_survival_factor(self, age: int, remaining_periods: int) -> float:
        return float(
            sum((self.cfg.discount**step) * self.survival_probability(age, step) for step in range(1, remaining_periods + 1))
        )

    def one_step_survival_bound(self, age: int) -> float:
        next_age = age + 1
        if next_age >= self.cfg.max_patience:
            return 0.0
        hazard = min(
            self.cfg.abandon_base + self.cfg.abandon_slope * next_age / max(self.cfg.max_patience, 1),
            1.0,
        )
        return max(1.0 - hazard, 0.0)

    def geometric_survival_factor(self, age: int) -> float:
        survival = self.one_step_survival_bound(age)
        ratio = self.cfg.discount * survival
        if ratio >= 1.0:
            return float("inf")
        return ratio / (1.0 - ratio)

    def arrival_geometric_survival_factor(self, age: int, arrival_prob: float) -> float:
        survival = self.one_step_survival_bound(age)
        ratio = self.cfg.discount * survival
        if ratio >= 1.0:
            return float("inf")
        delayed_ratio = ratio * max(1.0 - arrival_prob, 0.0)
        if delayed_ratio >= 1.0:
            return float("inf")
        return ratio / (1.0 - ratio) - delayed_ratio / (1.0 - delayed_ratio)

    def primitive_option_bound(
        self,
        buyers: SideState,
        sellers: SideState,
        t: int,
        efficient_k: int,
        mcafee_k: int,
    ) -> float:
        b_sorted = canonical_buyers(buyers)
        s_sorted = canonical_sellers(sellers)
        buyer_idx, buyer_age = b_sorted[efficient_k - 1]
        seller_idx, seller_age = s_sorted[efficient_k - 1]
        remaining_periods = max(self.cfg.horizon - t - 1, 0)
        value_min = min(self.cfg.values)
        value_max = max(self.cfg.values)
        value_range = value_max - value_min
        buyer_factor = self.discounted_survival_factor(buyer_age, remaining_periods)
        seller_factor = self.discounted_survival_factor(seller_age, remaining_periods)
        return value_range * (buyer_factor + seller_factor)

    def geometric_option_bound(
        self,
        buyers: SideState,
        sellers: SideState,
        efficient_k: int,
    ) -> float:
        b_sorted = canonical_buyers(buyers)
        s_sorted = canonical_sellers(sellers)
        _, buyer_age = b_sorted[efficient_k - 1]
        _, seller_age = s_sorted[efficient_k - 1]
        value_min = min(self.cfg.values)
        value_max = max(self.cfg.values)
        value_range = value_max - value_min
        buyer_factor = self.geometric_survival_factor(buyer_age)
        seller_factor = self.geometric_survival_factor(seller_age)
        return value_range * (buyer_factor + seller_factor)

    def arrival_geometric_option_bound(
        self,
        buyers: SideState,
        sellers: SideState,
        efficient_k: int,
    ) -> float:
        b_sorted = canonical_buyers(buyers)
        s_sorted = canonical_sellers(sellers)
        _, buyer_age = b_sorted[efficient_k - 1]
        _, seller_age = s_sorted[efficient_k - 1]
        value_min = min(self.cfg.values)
        value_max = max(self.cfg.values)
        value_range = value_max - value_min
        buyer_factor = self.arrival_geometric_survival_factor(buyer_age, self.cfg.arrival_prob_seller)
        seller_factor = self.arrival_geometric_survival_factor(seller_age, self.cfg.arrival_prob_buyer)
        return value_range * (buyer_factor + seller_factor)

    @lru_cache(maxsize=None)
    def survival_outcomes_side(self, side: SideState, is_buyer: bool) -> tuple[tuple[SideState, float], ...]:
        canonical = canonical_buyers if is_buyer else canonical_sellers
        outcomes: dict[SideState, float] = {(): 1.0}
        for idx, age in side:
            new_age = age + 1
            abandon_prob = 1.0 if new_age >= self.cfg.max_patience else min(
                self.cfg.abandon_base + self.cfg.abandon_slope * new_age / max(self.cfg.max_patience, 1),
                1.0,
            )
            next_outcomes: DefaultDict[SideState, float] = defaultdict(float)
            for kept, prob in outcomes.items():
                next_outcomes[kept] += prob * abandon_prob
                if abandon_prob < 1.0:
                    next_outcomes[canonical(kept + ((idx, new_age),))] += prob * (1.0 - abandon_prob)
            outcomes = dict(next_outcomes)
        return tuple(outcomes.items())

    def post_trade_transitions(self, buyers: SideState, sellers: SideState) -> tuple[tuple[FullState, float], ...]:
        outcomes: DefaultDict[FullState, float] = defaultdict(float)
        for next_b, prob_b in self.survival_outcomes_side(buyers, True):
            for next_s, prob_s in self.survival_outcomes_side(sellers, False):
                outcomes[(next_b, next_s)] += prob_b * prob_s
        return tuple(outcomes.items())

    @lru_cache(maxsize=None)
    def expected_next_value(self, buyers: SideState, sellers: SideState, t: int, policy: str) -> float:
        total = 0.0
        for next_state, prob in self.post_trade_transitions(buyers, sellers):
            total += prob * self.V(t + 1, next_state[0], next_state[1], policy)
        return total

    @lru_cache(maxsize=None)
    def V(self, t: int, buyers: SideState, sellers: SideState, policy: str) -> float:
        if t >= self.cfg.horizon:
            return 0.0
        total = 0.0
        for arrived_b, prob_b in self.arrival_outcomes(buyers, True):
            for arrived_s, prob_s in self.arrival_outcomes(sellers, False):
                prob = prob_b * prob_s
                k = self.trade_count(arrived_b, arrived_s, t, policy)
                remaining_b, remaining_s = self.remove_trades(arrived_b, arrived_s, k)
                surplus = self.surplus_for_count(arrived_b, arrived_s, k)
                wait_penalty = self.cfg.wait_cost * (len(remaining_b) + len(remaining_s))
                continuation = self.expected_next_value(remaining_b, remaining_s, t, policy)
                total += prob * (surplus - wait_penalty + self.cfg.discount * continuation)
        return total

    def relaxation_terms(self, buyers: SideState, sellers: SideState, t: int, future_policy: str) -> dict[str, float] | None:
        efficient_k = self.efficient_count(buyers, sellers)
        mcafee_k = self.mcafee_count(buyers, sellers)
        if efficient_k <= mcafee_k:
            return None
        clear_b, clear_s = self.remove_trades(buyers, sellers, efficient_k)
        leave_b, leave_s = self.remove_trades(buyers, sellers, mcafee_k)
        surplus_gain = self.surplus_for_count(buyers, sellers, efficient_k) - self.surplus_for_count(
            buyers, sellers, mcafee_k
        )
        wait_saving = self.cfg.wait_cost * ((len(leave_b) + len(leave_s)) - (len(clear_b) + len(clear_s)))
        future_gain = self.cfg.discount * (
            self.expected_next_value(clear_b, clear_s, t, future_policy)
            - self.expected_next_value(leave_b, leave_s, t, future_policy)
        )
        total_gain = surplus_gain + wait_saving + future_gain
        ages = [age for _, age in buyers + sellers]
        mean_age = float(np.mean(ages)) if ages else 0.0
        imbalance = abs(len(buyers) - len(sellers)) / max(self.cfg.max_buyers + self.cfg.max_sellers, 1)
        primitive_bound = self.primitive_option_bound(buyers, sellers, t, efficient_k, mcafee_k)
        geometric_bound = self.geometric_option_bound(buyers, sellers, efficient_k)
        arrival_geometric_bound = self.arrival_geometric_option_bound(buyers, sellers, efficient_k)
        primitive_certified = surplus_gain + wait_saving >= primitive_bound - 1.0e-12
        geometric_certified = surplus_gain + wait_saving >= geometric_bound - 1.0e-12
        arrival_geometric_certified = surplus_gain + wait_saving >= arrival_geometric_bound - 1.0e-12
        b_sorted = canonical_buyers(buyers)
        s_sorted = canonical_sellers(sellers)
        marginal_buyer_idx, marginal_buyer_age = b_sorted[efficient_k - 1]
        marginal_seller_idx, marginal_seller_age = s_sorted[efficient_k - 1]
        return {
            "efficient_k": float(efficient_k),
            "mcafee_k": float(mcafee_k),
            "relaxed_by_rule": float(self.queue_aware_count(buyers, sellers, t) > mcafee_k),
            "marginal_buyer_value": self.value(marginal_buyer_idx),
            "marginal_seller_cost": self.value(marginal_seller_idx),
            "marginal_buyer_age": float(marginal_buyer_age),
            "marginal_seller_age": float(marginal_seller_age),
            "marginal_surplus": surplus_gain,
            "current_wait_saving": wait_saving,
            "future_clearing_gain": future_gain,
            "future_option_value": -future_gain,
            "primitive_option_bound": primitive_bound,
            "primitive_certified": float(primitive_certified),
            "geometric_option_bound": geometric_bound,
            "geometric_certified": float(geometric_certified),
            "arrival_geometric_option_bound": arrival_geometric_bound,
            "arrival_geometric_certified": float(arrival_geometric_certified),
            "required_spread": -(wait_saving + future_gain),
            "total_gain": total_gain,
            "welfare_improving": float(total_gain >= -1.0e-12),
            "mean_age": mean_age,
            "imbalance": imbalance,
            "queue_length": float(len(buyers) + len(sellers)),
            "late_market": float(
                float(t) / max(self.cfg.horizon - 1, 1)
                >= 1.0 - self.cfg.terminal_window / max(self.cfg.horizon - 1, 1)
            ),
            "congested": float(self.is_congested(buyers, sellers, t)),
        }

    def forward_pretrade_distributions(self, policy: str) -> list[dict[FullState, float]]:
        start_dist: dict[FullState, float] = {((), ()): 1.0}
        pretrade_dists: list[dict[FullState, float]] = []
        for t in range(self.cfg.horizon):
            pretrade: DefaultDict[FullState, float] = defaultdict(float)
            next_start: DefaultDict[FullState, float] = defaultdict(float)
            for state, state_prob in start_dist.items():
                buyers, sellers = state
                for arrived_b, prob_b in self.arrival_outcomes(buyers, True):
                    for arrived_s, prob_s in self.arrival_outcomes(sellers, False):
                        prob = state_prob * prob_b * prob_s
                        pre_state = (arrived_b, arrived_s)
                        pretrade[pre_state] += prob
                        k = self.trade_count(arrived_b, arrived_s, t, policy)
                        remaining_b, remaining_s = self.remove_trades(arrived_b, arrived_s, k)
                        for next_state, trans_prob in self.post_trade_transitions(remaining_b, remaining_s):
                            next_start[next_state] += prob * trans_prob
            pretrade_dists.append(dict(pretrade))
            start_dist = dict(next_start)
        return pretrade_dists


def weighted_mean(rows: list[dict[str, float]], weights: np.ndarray, key: str) -> float:
    values = np.array([row[key] for row in rows], dtype=float)
    return float(np.sum(weights * values) / np.sum(weights)) if np.sum(weights) > 0 else float("nan")


def summarize_scenario(name: str, cfg: DPConfig, future_policy: str) -> tuple[dict[str, float], pd.DataFrame]:
    dp = RelaxationDP(cfg)
    pretrade_dists = dp.forward_pretrade_distributions("mcafee")
    detail_rows: list[dict[str, float | str]] = []
    for t, dist in enumerate(pretrade_dists):
        for state, prob in dist.items():
            terms = dp.relaxation_terms(state[0], state[1], t, future_policy)
            if terms is None:
                continue
            detail_rows.append({"scenario": name, "t": t, "prob": prob, **terms})
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return {
            "scenario": name,
            "opportunity_mass": 0.0,
            "relaxed_mass": 0.0,
            "relaxed_share": float("nan"),
            "triggered_required_spread": float("nan"),
            "not_triggered_required_spread": float("nan"),
            "triggered_total_gain": float("nan"),
            "not_triggered_total_gain": float("nan"),
            "triggered_future_gain": float("nan"),
            "triggered_improve_share": float("nan"),
            "not_triggered_improve_share": float("nan"),
        }, detail

    opportunity_mass = float(detail["prob"].sum())
    triggered = detail[detail["relaxed_by_rule"] > 0.5]
    not_triggered = detail[detail["relaxed_by_rule"] < 0.5]

    def group_mean(df: pd.DataFrame, col: str) -> float:
        if df.empty:
            return float("nan")
        return float(np.average(df[col].to_numpy(dtype=float), weights=df["prob"].to_numpy(dtype=float)))

    relaxed_mass = float(triggered["prob"].sum()) if not triggered.empty else 0.0
    summary = {
        "scenario": name,
        "opportunity_mass": opportunity_mass,
        "relaxed_mass": relaxed_mass,
        "relaxed_share": relaxed_mass / opportunity_mass if opportunity_mass else float("nan"),
        "triggered_required_spread": group_mean(triggered, "required_spread"),
        "not_triggered_required_spread": group_mean(not_triggered, "required_spread"),
        "trigger_required_spread_gap": group_mean(not_triggered, "required_spread")
        - group_mean(triggered, "required_spread"),
        "triggered_marginal_surplus": group_mean(triggered, "marginal_surplus"),
        "triggered_current_wait_saving": group_mean(triggered, "current_wait_saving"),
        "triggered_future_gain": group_mean(triggered, "future_clearing_gain"),
        "triggered_total_gain": group_mean(triggered, "total_gain"),
        "not_triggered_total_gain": group_mean(not_triggered, "total_gain"),
        "trigger_total_gain_gap": group_mean(triggered, "total_gain") - group_mean(not_triggered, "total_gain"),
        "triggered_improve_share": group_mean(triggered, "welfare_improving"),
        "not_triggered_improve_share": group_mean(not_triggered, "welfare_improving"),
        "trigger_improve_share_gap": group_mean(triggered, "welfare_improving")
        - group_mean(not_triggered, "welfare_improving"),
        "triggered_bound": group_mean(triggered, "primitive_option_bound"),
        "triggered_certified_share": group_mean(triggered, "primitive_certified"),
        "certified_success_share": group_mean(
            triggered[triggered["primitive_certified"] > 0.5],
            "welfare_improving",
        ),
        "triggered_geo_bound": group_mean(triggered, "geometric_option_bound"),
        "triggered_geo_certified_share": group_mean(triggered, "geometric_certified"),
        "geo_certified_success_share": group_mean(
            triggered[triggered["geometric_certified"] > 0.5],
            "welfare_improving",
        ),
        "triggered_arrival_geo_bound": group_mean(triggered, "arrival_geometric_option_bound"),
        "triggered_arrival_geo_certified_share": group_mean(triggered, "arrival_geometric_certified"),
        "arrival_geo_certified_success_share": group_mean(
            triggered[triggered["arrival_geometric_certified"] > 0.5],
            "welfare_improving",
        ),
        "triggered_mean_age": group_mean(triggered, "mean_age"),
        "triggered_imbalance": group_mean(triggered, "imbalance"),
    }
    return summary, detail


def scenario_configs(base: DPConfig) -> list[tuple[str, DPConfig]]:
    return [
        ("Baseline", base),
        ("Low friction", replace(base, wait_cost=0.005, abandon_slope=0.05)),
        ("High wait cost", replace(base, wait_cost=0.040)),
        ("Impatient", replace(base, max_patience=3, abandon_slope=0.25)),
        ("Sparse arrivals", replace(base, arrival_prob_buyer=0.35, arrival_prob_seller=0.35)),
        ("High arrivals", replace(base, arrival_prob_buyer=0.90, arrival_prob_seller=0.90)),
    ]


def write_tex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Scenario & Req. & Fin. cert. & Geo. bd. & Arr.-geo bd. & Arr.-geo cert. & Gain & Improve \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['scenario']} & {row['triggered_required_spread']:.3f} & "
            f"{row['triggered_certified_share']:.2f} & {row['triggered_geo_bound']:.2f} & "
            f"{row['triggered_arrival_geo_bound']:.2f} & {row['triggered_arrival_geo_certified_share']:.2f} & "
            f"{row['triggered_total_gain']:.3f} & "
            f"{row['triggered_improve_share']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trigger_validation_tex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Scenario & Trig. share & Req. T & Req. N & Gain T & Gain N & Improve gap \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['scenario']} & {row['relaxed_share']:.2f} & "
            f"{row['triggered_required_spread']:.3f} & {row['not_triggered_required_spread']:.3f} & "
            f"{row['triggered_total_gain']:.3f} & {row['not_triggered_total_gain']:.3f} & "
            f"{row['trigger_improve_share_gap']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", default="0.2,0.5,0.8")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--future-policy", default="mcafee", choices=["mcafee", "first_best", "queue_aware"])
    parser.add_argument("--out-root", default="experiments/dynamic_relaxation_thresholds")
    parser.add_argument("--paper-table", default="paper/tables/dynamic_relaxation_thresholds.tex")
    parser.add_argument("--trigger-table", default="paper/tables/queue_trigger_validation.tex")
    args = parser.parse_args()

    base = DPConfig(values=tuple(float(x) for x in args.values.split(",") if x.strip()), horizon=args.horizon)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float]] = []
    details: list[pd.DataFrame] = []
    for name, cfg in scenario_configs(base):
        summary, detail = summarize_scenario(name, cfg, args.future_policy)
        summaries.append(summary)
        if not detail.empty:
            detail["future_policy"] = args.future_policy
            details.append(detail)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_df.to_csv(out_root / "dynamic_relaxation_thresholds.csv", index=False)
    detail_df.to_csv(out_root / "dynamic_relaxation_threshold_details.csv", index=False)
    write_tex(summary_df, Path(args.paper_table))
    write_trigger_validation_tex(summary_df, Path(args.trigger_table))
    manifest = {
        "base_config": asdict(base),
        "future_policy": args.future_policy,
        "summary": str(out_root / "dynamic_relaxation_thresholds.csv"),
        "details": str(out_root / "dynamic_relaxation_threshold_details.csv"),
        "paper_table": args.paper_table,
        "trigger_table": args.trigger_table,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
