from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from dynamic_relaxation_thresholds import DPConfig
from exact_dynamic_audit import summarize_exact_audit


COARSE_REPORTS = (0.0, 0.2, 0.5, 0.8, 1.0)
DENSE_REPORTS = tuple(round(0.1 * i, 1) for i in range(11))


def scenario_configs() -> list[tuple[str, DPConfig, tuple[float, ...]]]:
    base = DPConfig()
    return [
        ("Baseline", base, COARSE_REPORTS),
        ("Finer value grid", replace(base, values=(0.2, 0.4, 0.6, 0.8)), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        ("Dense reports", base, DENSE_REPORTS),
        ("Horizon 6", replace(base, horizon=6), COARSE_REPORTS),
        ("Horizon 7", replace(base, horizon=7), COARSE_REPORTS),
        ("High wait cost", replace(base, wait_cost=0.040), COARSE_REPORTS),
        ("Impatient", replace(base, max_patience=3, abandon_slope=0.25), COARSE_REPORTS),
        (
            "Sparse arrivals",
            replace(base, arrival_prob_buyer=0.35, arrival_prob_seller=0.35),
            COARSE_REPORTS,
        ),
    ]


def build_robustness_table() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, float | int | str]] = []
    details: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []

    for scenario, cfg, report_grid in scenario_configs():
        summary, detail = summarize_exact_audit(cfg, report_grid)
        detail = detail.copy()
        detail.insert(0, "scenario", scenario)
        details.append(detail)

        mcafee = summary.loc[summary["policy"] == "McAfee"].iloc[0]
        queue = summary.loc[summary["policy"] == "Queue aware"].iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "horizon": cfg.horizon,
                "value_grid_size": len(cfg.values),
                "report_grid_size": len(report_grid),
                "mcafee_objective": float(mcafee["objective"]),
                "queue_objective": float(queue["objective"]),
                "objective_gap": float(queue["objective"] - mcafee["objective"]),
                "myopic_fb_ratio_gap": float(queue["myopic_fb_ratio"] - mcafee["myopic_fb_ratio"]),
                "br_mean_gap": float(queue["exact_no_exit_mean"] - mcafee["exact_no_exit_mean"]),
                "br_max_gap": float(queue["exact_no_exit_max"] - mcafee["exact_no_exit_max"]),
                "exit_mean_gap": float(queue["exact_exit_mean"] - mcafee["exact_exit_mean"]),
                "exit_max_gap": float(queue["exact_exit_max"] - mcafee["exact_exit_max"]),
                "queue_exit_max": float(queue["exact_exit_max"]),
            }
        )
        manifests.append({"scenario": scenario, "config": asdict(cfg), "report_grid": report_grid})

    return pd.DataFrame(rows), pd.concat(details, ignore_index=True), manifests


def write_tex(robustness: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Scenario & M obj. & Q obj. & Obj. gap & BR mean gap & Exit mean gap & Q exit max \\\\",
        "\\midrule",
    ]
    for _, row in robustness.iterrows():
        lines.append(
            f"{row['scenario']} & {row['mcafee_objective']:.3f} & "
            f"{row['queue_objective']:.3f} & {row['objective_gap']:.3f} & "
            f"{row['br_mean_gap']:.3f} & "
            f"{row['exit_mean_gap']:.3f} & {row['queue_exit_max']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="experiments/exact_dynamic_robustness")
    parser.add_argument("--paper-table", default="paper/tables/exact_dynamic_robustness.tex")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    robustness, detail, manifests = build_robustness_table()
    robustness.to_csv(out_root / "exact_dynamic_robustness_summary.csv", index=False)
    detail.to_csv(out_root / "exact_dynamic_robustness_detail.csv", index=False)
    write_tex(robustness, Path(args.paper_table))
    (out_root / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
