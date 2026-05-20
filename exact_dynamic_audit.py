from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Literal

import numpy as np
import pandas as pd

from dynamic_relaxation_thresholds import DPConfig, RelaxationDP, SideState, canonical_buyers, canonical_sellers


Side = Literal["buyer", "seller"]
Policy = Literal["mcafee", "queue_aware"]


class ExactTaggedAudit:
    """Exact finite-state Markov report/exit audit for one tagged entrant.

    The audit is intentionally small. Other agents report truthfully. The tagged
    agent can choose a report from a finite grid in every period and can also
    exit before trade when exit is enabled. Future values are computed exactly
    over the finite arrival/abandonment process.
    """

    def __init__(self, cfg: DPConfig, report_grid: tuple[float, ...]) -> None:
        self.cfg = cfg
        self.report_grid = tuple(sorted(set(report_grid)))
        self.social_dp = RelaxationDP(cfg)

    def value(self, idx: int) -> float:
        return self.cfg.values[idx]

    def canonical(self, side: SideState, is_buyer: bool) -> SideState:
        return canonical_buyers(side) if is_buyer else canonical_sellers(side)

    @lru_cache(maxsize=None)
    def arrival_outcomes_with_cap(
        self,
        side: SideState,
        is_buyer: bool,
        cap: int,
    ) -> tuple[tuple[SideState, float], ...]:
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

    @lru_cache(maxsize=None)
    def survival_outcomes_side(self, side: SideState, is_buyer: bool) -> tuple[tuple[SideState, float], ...]:
        return self.social_dp.survival_outcomes_side(side, is_buyer)

    def active_rows(
        self,
        side: Side,
        tag_idx: int,
        tag_age: int,
        other_buyers: SideState,
        other_sellers: SideState,
        report: float,
    ) -> tuple[list[dict[str, float | bool | str | int]], list[dict[str, float | bool | str | int]]]:
        buyers: list[dict[str, float | bool | str | int]] = [
            {"type": idx, "age": age, "report": self.value(idx), "tag": False, "side": "buyer"}
            for idx, age in other_buyers
        ]
        sellers: list[dict[str, float | bool | str | int]] = [
            {"type": idx, "age": age, "report": self.value(idx), "tag": False, "side": "seller"}
            for idx, age in other_sellers
        ]
        if side == "buyer":
            buyers.append({"type": tag_idx, "age": tag_age, "report": report, "tag": True, "side": "buyer"})
        else:
            sellers.append({"type": tag_idx, "age": tag_age, "report": report, "tag": True, "side": "seller"})
        return buyers, sellers

    def padded_sorted_reports(
        self,
        buyers: list[dict[str, float | bool | str | int]],
        sellers: list[dict[str, float | bool | str | int]],
    ) -> tuple[list[dict[str, float | bool | str | int]], list[dict[str, float | bool | str | int]]]:
        padded_buyers = list(buyers)
        padded_sellers = list(sellers)
        for _ in range(self.cfg.max_buyers - len(padded_buyers)):
            padded_buyers.append({"type": -1, "age": 0, "report": 0.0, "tag": False, "side": "buyer", "inactive": True})
        for _ in range(self.cfg.max_sellers - len(padded_sellers)):
            padded_sellers.append({"type": -1, "age": 0, "report": 1.0, "tag": False, "side": "seller", "inactive": True})
        padded_buyers.sort(key=lambda row: (-float(row["report"]), bool(row.get("inactive", False)), bool(row["tag"])))
        padded_sellers.sort(key=lambda row: (float(row["report"]), bool(row.get("inactive", False)), bool(row["tag"])))
        return padded_buyers, padded_sellers

    def efficient_count_from_reports(
        self,
        sorted_buyers: list[dict[str, float | bool | str | int]],
        sorted_sellers: list[dict[str, float | bool | str | int]],
    ) -> int:
        return sum(float(b["report"]) >= float(s["report"]) for b, s in zip(sorted_buyers, sorted_sellers))

    def mcafee_count_from_reports(
        self,
        sorted_buyers: list[dict[str, float | bool | str | int]],
        sorted_sellers: list[dict[str, float | bool | str | int]],
    ) -> int:
        efficient_k = self.efficient_count_from_reports(sorted_buyers, sorted_sellers)
        if efficient_k == 0:
            return 0
        trade_k = efficient_k - 1
        if efficient_k < self.cfg.max_buyers and efficient_k < self.cfg.max_sellers:
            price = 0.5 * (float(sorted_buyers[efficient_k]["report"]) + float(sorted_sellers[efficient_k]["report"]))
            if float(sorted_sellers[efficient_k - 1]["report"]) <= price <= float(sorted_buyers[efficient_k - 1]["report"]):
                trade_k = efficient_k
        return max(trade_k, 0)

    def is_congested(
        self,
        buyers: list[dict[str, float | bool | str | int]],
        sellers: list[dict[str, float | bool | str | int]],
        t: int,
    ) -> bool:
        ages = [int(row["age"]) for row in buyers + sellers if not bool(row.get("inactive", False))]
        mean_age = float(np.mean(ages)) if ages else 0.0
        imbalance = abs(len(buyers) - len(sellers)) / max(self.cfg.max_buyers + self.cfg.max_sellers, 1)
        late_market = float(t) / max(self.cfg.horizon - 1, 1) >= 1.0 - self.cfg.terminal_window / max(
            self.cfg.horizon - 1,
            1,
        )
        return (
            mean_age >= self.cfg.age_trigger * self.cfg.max_patience
            or imbalance >= self.cfg.imbalance_trigger
            or late_market
        )

    def trade_count(
        self,
        policy: Policy,
        buyers: list[dict[str, float | bool | str | int]],
        sellers: list[dict[str, float | bool | str | int]],
        sorted_buyers: list[dict[str, float | bool | str | int]],
        sorted_sellers: list[dict[str, float | bool | str | int]],
        t: int,
    ) -> int:
        efficient_k = self.efficient_count_from_reports(sorted_buyers, sorted_sellers)
        mcafee_k = self.mcafee_count_from_reports(sorted_buyers, sorted_sellers)
        if policy == "mcafee" or efficient_k <= mcafee_k:
            return mcafee_k
        return efficient_k if self.is_congested(buyers, sellers, t) else mcafee_k

    def outcome(
        self,
        side: Side,
        tag_idx: int,
        tag_age: int,
        other_buyers: SideState,
        other_sellers: SideState,
        t: int,
        policy: Policy,
        report: float,
    ) -> tuple[float, bool, SideState, SideState]:
        buyers, sellers = self.active_rows(side, tag_idx, tag_age, other_buyers, other_sellers, report)
        sorted_buyers, sorted_sellers = self.padded_sorted_reports(buyers, sellers)
        k = self.trade_count(policy, buyers, sellers, sorted_buyers, sorted_sellers, t)
        if k <= 0:
            period_utility = -self.cfg.wait_cost
            return period_utility, False, canonical_buyers(other_buyers), canonical_sellers(other_sellers)

        last_b = float(sorted_buyers[k - 1]["report"])
        last_s = float(sorted_sellers[k - 1]["report"])
        if k < self.cfg.max_buyers and k < self.cfg.max_sellers:
            price = 0.5 * (float(sorted_buyers[k]["report"]) + float(sorted_sellers[k]["report"]))
            price = min(max(price, last_s), last_b)
        else:
            price = 0.5 * (last_b + last_s)

        traded_buyers = sorted_buyers[:k]
        traded_sellers = sorted_sellers[:k]
        tag_matched = any(bool(row["tag"]) for row in traded_buyers + traded_sellers)
        if side == "buyer":
            period_utility = self.value(tag_idx) - price if tag_matched else -self.cfg.wait_cost
        else:
            period_utility = price - self.value(tag_idx) if tag_matched else -self.cfg.wait_cost

        remaining_buyers: list[tuple[int, int]] = []
        remaining_sellers: list[tuple[int, int]] = []
        for row in sorted_buyers[k:]:
            if bool(row.get("inactive", False)) or bool(row["tag"]):
                continue
            remaining_buyers.append((int(row["type"]), int(row["age"])))
        for row in sorted_sellers[k:]:
            if bool(row.get("inactive", False)) or bool(row["tag"]):
                continue
            remaining_sellers.append((int(row["type"]), int(row["age"])))
        return period_utility, tag_matched, canonical_buyers(tuple(remaining_buyers)), canonical_sellers(tuple(remaining_sellers))

    @lru_cache(maxsize=None)
    def truthful_value(
        self,
        t: int,
        side: Side,
        tag_idx: int,
        tag_age: int,
        other_buyers: SideState,
        other_sellers: SideState,
        policy: Policy,
    ) -> float:
        return self.action_value(
            t,
            side,
            tag_idx,
            tag_age,
            other_buyers,
            other_sellers,
            policy,
            self.value(tag_idx),
            False,
            "truthful",
        )

    @lru_cache(maxsize=None)
    def best_value(
        self,
        t: int,
        side: Side,
        tag_idx: int,
        tag_age: int,
        other_buyers: SideState,
        other_sellers: SideState,
        policy: Policy,
        allow_exit: bool,
    ) -> float:
        if t >= self.cfg.horizon:
            return 0.0
        candidates = [
            self.action_value(
                t,
                side,
                tag_idx,
                tag_age,
                other_buyers,
                other_sellers,
                policy,
                r,
                allow_exit,
                "best",
            )
            for r in self.report_grid
        ]
        if allow_exit:
            candidates.append(0.0)
        return max(candidates)

    def action_value(
        self,
        t: int,
        side: Side,
        tag_idx: int,
        tag_age: int,
        other_buyers: SideState,
        other_sellers: SideState,
        policy: Policy,
        report: float,
        allow_exit: bool,
        future_mode: Literal["truthful", "best"],
    ) -> float:
        if t >= self.cfg.horizon:
            return 0.0
        same_cap = (self.cfg.max_buyers - 1) if side == "buyer" else self.cfg.max_buyers
        seller_cap = self.cfg.max_sellers if side == "buyer" else (self.cfg.max_sellers - 1)
        total = 0.0
        for arrived_b, prob_b in self.arrival_outcomes_with_cap(other_buyers, True, same_cap):
            for arrived_s, prob_s in self.arrival_outcomes_with_cap(other_sellers, False, seller_cap):
                prob = prob_b * prob_s
                period_u, tag_matched, remaining_b, remaining_s = self.outcome(
                    side, tag_idx, tag_age, arrived_b, arrived_s, t, policy, report
                )
                if tag_matched:
                    total += prob * period_u
                    continue
                next_age = tag_age + 1
                abandon_prob = 1.0 if next_age >= self.cfg.max_patience else min(
                    self.cfg.abandon_base + self.cfg.abandon_slope * next_age / max(self.cfg.max_patience, 1),
                    1.0,
                )
                future = 0.0
                if abandon_prob < 1.0:
                    for survived_b, prob_surv_b in self.survival_outcomes_side(remaining_b, True):
                        for survived_s, prob_surv_s in self.survival_outcomes_side(remaining_s, False):
                            if future_mode == "truthful":
                                continuation_value = self.truthful_value(
                                    t + 1,
                                    side,
                                    tag_idx,
                                    next_age,
                                    survived_b,
                                    survived_s,
                                    policy,
                                )
                            else:
                                continuation_value = self.best_value(
                                    t + 1,
                                    side,
                                    tag_idx,
                                    next_age,
                                    survived_b,
                                    survived_s,
                                    policy,
                                    allow_exit,
                                )
                            future += (
                                (1.0 - abandon_prob)
                                * prob_surv_b
                                * prob_surv_s
                                * continuation_value
                            )
                total += prob * (period_u + self.cfg.discount * future)
        return total

    def entry_regrets(self, policy: Policy) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for side in ("buyer", "seller"):
            for idx in range(len(self.cfg.values)):
                other_buyers: SideState = ()
                other_sellers: SideState = ()
                truth = self.truthful_value(0, side, idx, 0, other_buyers, other_sellers, policy)
                best_no_exit = self.best_value(0, side, idx, 0, other_buyers, other_sellers, policy, False)
                best_exit = self.best_value(0, side, idx, 0, other_buyers, other_sellers, policy, True)
                rows.append(
                    {
                        "policy": policy,
                        "side": side,
                        "type": self.value(idx),
                        "truthful_value": truth,
                        "best_no_exit_value": best_no_exit,
                        "best_exit_value": best_exit,
                        "exact_no_exit_regret": max(best_no_exit - truth, 0.0),
                        "exact_exit_regret": max(best_exit - truth, 0.0),
                    }
                )
        return rows


def summarize_exact_audit(cfg: DPConfig, report_grid: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = ExactTaggedAudit(cfg, report_grid)
    detail_rows: list[dict[str, float | str]] = []
    for policy in ("mcafee", "queue_aware"):
        detail_rows.extend(audit.entry_regrets(policy))
    detail = pd.DataFrame(detail_rows)

    social_rows = []
    first_best = audit.social_dp.V(0, (), (), "first_best")
    for policy in ("mcafee", "queue_aware"):
        policy_detail = detail[detail["policy"] == policy]
        obj = audit.social_dp.V(0, (), (), policy)
        social_rows.append(
            {
                "policy": "McAfee" if policy == "mcafee" else "Queue aware",
                "objective": obj,
                "myopic_fb_ratio": obj / first_best if first_best else float("nan"),
                "exact_no_exit_mean": float(policy_detail["exact_no_exit_regret"].mean()),
                "exact_no_exit_max": float(policy_detail["exact_no_exit_regret"].max()),
                "exact_exit_mean": float(policy_detail["exact_exit_regret"].mean()),
                "exact_exit_max": float(policy_detail["exact_exit_regret"].max()),
            }
        )
    summary = pd.DataFrame(social_rows)
    return summary, detail


def write_tex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Policy & Obj. & Myopic-FB & BR mean & BR max & Exit mean & Exit max \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['policy']} & {row['objective']:.3f} & {row['myopic_fb_ratio']:.3f} & "
            f"{row['exact_no_exit_mean']:.3f} & {row['exact_no_exit_max']:.3f} & "
            f"{row['exact_exit_mean']:.3f} & {row['exact_exit_max']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", default="0.2,0.5,0.8")
    parser.add_argument("--reports", default="0.0,0.2,0.5,0.8,1.0")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--out-root", default="experiments/exact_dynamic_audit")
    parser.add_argument("--paper-table", default="paper/tables/exact_dynamic_audit.tex")
    args = parser.parse_args()

    cfg = DPConfig(values=tuple(float(x) for x in args.values.split(",") if x.strip()), horizon=args.horizon)
    report_grid = tuple(float(x) for x in args.reports.split(",") if x.strip())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary, detail = summarize_exact_audit(cfg, report_grid)
    summary.to_csv(out_root / "exact_dynamic_audit_summary.csv", index=False)
    detail.to_csv(out_root / "exact_dynamic_audit_detail.csv", index=False)
    write_tex(summary, Path(args.paper_table))
    manifest = {
        "config": asdict(cfg),
        "report_grid": report_grid,
        "summary": str(out_root / "exact_dynamic_audit_summary.csv"),
        "detail": str(out_root / "exact_dynamic_audit_detail.csv"),
        "paper_table": args.paper_table,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
