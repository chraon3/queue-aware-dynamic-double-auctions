from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_penalty_table(stress_csv: Path) -> pd.DataFrame:
    stress = pd.read_csv(stress_csv)
    rows: list[dict[str, float | str]] = []
    for _, row in stress.iterrows():
        obj_gap = float(row["gap_objective"])
        abandon_gap = float(row["gap_abandonment"])
        lambda_star = obj_gap / abandon_gap if abandon_gap > 0 else float("inf")
        lambda_one_gap = obj_gap - abandon_gap
        rows.append(
            {
                "scenario": str(row["label"]),
                "objective_gap": obj_gap,
                "abandonment_gap": abandon_gap,
                "lambda_star": lambda_star,
                "net_gap_lambda_1": lambda_one_gap,
                "winner_lambda_1": "Queue" if lambda_one_gap >= 0 else "Neural",
            }
        )
    return pd.DataFrame(rows)


def write_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Scenario & Obj. gap & Aband. gap & $\\lambda^*$ & Gap at $\\lambda=1$ & Winner \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['scenario']} & {row['objective_gap']:.3f} & "
            f"{row['abandonment_gap']:.3f} & {row['lambda_star']:.2f} & "
            f"{row['net_gap_lambda_1']:.3f} & {row['winner_lambda_1']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-csv", default="paper/tables/queue_aware_stress.csv")
    parser.add_argument("--out-root", default="experiments/abandonment_penalty_sensitivity")
    parser.add_argument("--paper-table", default="paper/tables/abandonment_penalty_sensitivity.tex")
    args = parser.parse_args()

    stress_csv = Path(args.stress_csv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    table = build_penalty_table(stress_csv)
    table.to_csv(out_root / "abandonment_penalty_sensitivity.csv", index=False)
    write_tex(table, Path(args.paper_table))
    manifest = {
        "stress_csv": str(stress_csv),
        "paper_table": args.paper_table,
        "definition": "lambda_star = gap_objective / gap_abandonment for W_lambda = W - lambda A",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
