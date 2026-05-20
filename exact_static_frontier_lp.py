from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from econpinn_double_auction import (
    first_best_welfare_np,
    mcafee_welfare_np,
    posted_price_welfare_np,
    trade_reduction_welfare_np,
)


def profile_grid(values: list[float], n_buyers: int, n_sellers: int) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    profiles = []
    for buyers in itertools.product(values, repeat=n_buyers):
        for sellers in itertools.product(values, repeat=n_sellers):
            profiles.append((tuple(buyers), tuple(sellers)))
    return profiles


class Indexer:
    def __init__(self, n_profiles: int, n_buyers: int, n_sellers: int) -> None:
        self.n_profiles = n_profiles
        self.n_buyers = n_buyers
        self.n_sellers = n_sellers
        self.x0 = 0
        self.p0 = self.x0 + n_profiles * n_buyers * n_sellers
        self.t0 = self.p0 + n_profiles * n_buyers
        self.n_vars = self.t0 + n_profiles * n_sellers

    def x(self, r: int, i: int, j: int) -> int:
        return self.x0 + (r * self.n_buyers + i) * self.n_sellers + j

    def p(self, r: int, i: int) -> int:
        return self.p0 + r * self.n_buyers + i

    def t(self, r: int, j: int) -> int:
        return self.t0 + r * self.n_sellers + j


def add_row(rows: list[np.ndarray], rhs: list[float], idx: Indexer, entries: list[tuple[int, float]], bound: float) -> None:
    row = np.zeros(idx.n_vars)
    for col, value in entries:
        row[col] += value
    rows.append(row)
    rhs.append(bound)


def solve_frontier_point(values: list[float], n_buyers: int, n_sellers: int, epsilon: float) -> dict[str, float]:
    profiles = profile_grid(values, n_buyers, n_sellers)
    profile_to_idx = {profile: idx for idx, profile in enumerate(profiles)}
    idx = Indexer(len(profiles), n_buyers, n_sellers)
    prob = 1.0 / len(profiles)

    c = np.zeros(idx.n_vars)
    bounds: list[tuple[float, float | None]] = [(0.0, 1.0)] * idx.n_vars

    for r, (buyers, sellers) in enumerate(profiles):
        for i in range(n_buyers):
            for j in range(n_sellers):
                col = idx.x(r, i, j)
                c[col] = -prob * (buyers[i] - sellers[j])
                if buyers[i] < sellers[j]:
                    bounds[col] = (0.0, 0.0)
        for i in range(n_buyers):
            bounds[idx.p(r, i)] = (0.0, 2.0)
        for j in range(n_sellers):
            bounds[idx.t(r, j)] = (0.0, 2.0)

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for r, (buyers, sellers) in enumerate(profiles):
        for i in range(n_buyers):
            add_row(rows, rhs, idx, [(idx.x(r, i, j), 1.0) for j in range(n_sellers)], 1.0)
        for j in range(n_sellers):
            add_row(rows, rhs, idx, [(idx.x(r, i, j), 1.0) for i in range(n_buyers)], 1.0)

        add_row(
            rows,
            rhs,
            idx,
            [(idx.t(r, j), 1.0) for j in range(n_sellers)] + [(idx.p(r, i), -1.0) for i in range(n_buyers)],
            0.0,
        )

        for i in range(n_buyers):
            add_row(
                rows,
                rhs,
                idx,
                [(idx.p(r, i), 1.0)] + [(idx.x(r, i, j), -buyers[i]) for j in range(n_sellers)],
                0.0,
            )
        for j in range(n_sellers):
            add_row(
                rows,
                rhs,
                idx,
                [(idx.x(r, i, j), sellers[j]) for i in range(n_buyers)] + [(idx.t(r, j), -1.0)],
                0.0,
            )

    for r, (buyers, sellers) in enumerate(profiles):
        for i in range(n_buyers):
            truth_entries = [(idx.x(r, i, j), -buyers[i]) for j in range(n_sellers)] + [(idx.p(r, i), 1.0)]
            for misreport in values:
                mis_buyers = list(buyers)
                mis_buyers[i] = misreport
                r_mis = profile_to_idx[(tuple(mis_buyers), sellers)]
                entries = truth_entries + [(idx.x(r_mis, i, j), buyers[i]) for j in range(n_sellers)] + [(idx.p(r_mis, i), -1.0)]
                add_row(rows, rhs, idx, entries, epsilon)

        for j in range(n_sellers):
            truth_entries = [(idx.t(r, j), -1.0)] + [(idx.x(r, i, j), sellers[j]) for i in range(n_buyers)]
            for misreport in values:
                mis_sellers = list(sellers)
                mis_sellers[j] = misreport
                r_mis = profile_to_idx[(buyers, tuple(mis_sellers))]
                entries = truth_entries + [(idx.t(r_mis, j), 1.0)] + [(idx.x(r_mis, i, j), -sellers[j]) for i in range(n_buyers)]
                add_row(rows, rhs, idx, entries, epsilon)

    res = linprog(
        c,
        A_ub=np.vstack(rows),
        b_ub=np.array(rhs),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed for epsilon={epsilon}: {res.message}")

    welfare = -float(res.fun)
    buyers_np = np.array([p[0] for p in profiles], dtype=float)
    sellers_np = np.array([p[1] for p in profiles], dtype=float)
    first_best = float(first_best_welfare_np(buyers_np, sellers_np).mean())
    mcafee = float(mcafee_welfare_np(buyers_np, sellers_np).mean())
    posted = float(posted_price_welfare_np(buyers_np, sellers_np).mean())
    trade_reduction = float(trade_reduction_welfare_np(buyers_np, sellers_np).mean())

    return {
        "epsilon": epsilon,
        "exact_welfare": welfare,
        "exact_efficiency": welfare / first_best if first_best > 0 else np.nan,
        "first_best": first_best,
        "mcafee_efficiency": mcafee / first_best if first_best > 0 else np.nan,
        "posted_efficiency": posted / first_best if first_best > 0 else np.nan,
        "trade_reduction_efficiency": trade_reduction / first_best if first_best > 0 else np.nan,
        "n_profiles": len(profiles),
        "n_constraints": len(rhs),
        "n_variables": idx.n_vars,
    }


def write_tex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "$\\varepsilon$ & Exact & McAfee & Posted & Profiles \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['epsilon']:.3f} & {row['exact_efficiency']:.3f} & "
            f"{row['mcafee_efficiency']:.3f} & {row['posted_efficiency']:.3f} & "
            f"{int(row['n_profiles'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--n-buyers", type=int, default=2)
    parser.add_argument("--n-sellers", type=int, default=2)
    parser.add_argument("--epsilons", default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--out-root", default="experiments/exact_static_frontier")
    parser.add_argument("--paper-table", default="paper/tables/exact_static_frontier.tex")
    args = parser.parse_args()

    values = [float(x) for x in args.values.split(",") if x.strip()]
    epsilons = [float(x) for x in args.epsilons.split(",") if x.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [solve_frontier_point(values, args.n_buyers, args.n_sellers, eps) for eps in epsilons]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "exact_static_frontier.csv", index=False)
    write_tex(summary, Path(args.paper_table))
    manifest = {
        "values": values,
        "n_buyers": args.n_buyers,
        "n_sellers": args.n_sellers,
        "epsilons": epsilons,
        "output": str(out_root / "exact_static_frontier.csv"),
        "paper_table": args.paper_table,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
