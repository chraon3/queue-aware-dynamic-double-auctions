from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path("experiments/queue_aware_mcafee_certified_multiseed")
NEURAL_BY_SEED = Path("paper/tables/dynamic_results_by_seed.csv")


SUMMARY_COLUMNS = [
    "policy",
    "runs",
    "objective",
    "efficiency",
    "mean_regret",
    "p95_regret",
    "max_regret",
    "abandonment",
]


def load_queue_aware() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE_DIR.glob("queue_aware_mcafee_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = int(path.stem.replace("queue_aware_mcafee_seed", ""))
        rows.append(
            {
                "seed": seed,
                "policy": "Payment-audited queue-aware McAfee",
                "objective": data["dynamic_queue_mcafee_certified_objective"],
                "efficiency": data["dynamic_queue_mcafee_certified_efficiency"],
                "mean_regret": data["dynamic_queue_mcafee_certified_mean_regret"],
                "p95_regret": data["dynamic_queue_mcafee_certified_p95_regret"],
                "max_regret": data["dynamic_queue_mcafee_certified_max_regret"],
                "abandonment": data["mean_abandonment"],
            }
        )
    return pd.DataFrame(rows)


def load_neural() -> pd.DataFrame:
    data = pd.read_csv(NEURAL_BY_SEED)
    neural = data.loc[data["policy"] == "Neural dynamic mechanism"].copy()
    neural["seed"] = neural["run"].str.extract(r"seed(\d+)").astype(int)
    neural = neural.rename(
        columns={
            "regret": "mean_regret",
        }
    )
    neural["max_regret"] = float("nan")
    return neural[
        [
            "seed",
            "policy",
            "objective",
            "efficiency",
            "mean_regret",
            "p95_regret",
            "max_regret",
            "abandonment",
        ]
    ]


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in data.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "runs": float(len(group)),
                "objective": float(group["objective"].mean()),
                "objective_std": float(group["objective"].std(ddof=1)),
                "efficiency": float(group["efficiency"].mean()),
                "efficiency_std": float(group["efficiency"].std(ddof=1)),
                "mean_regret": float(group["mean_regret"].mean()),
                "mean_regret_std": float(group["mean_regret"].std(ddof=1)),
                "p95_regret": float(group["p95_regret"].mean()),
                "p95_regret_std": float(group["p95_regret"].std(ddof=1)),
                "max_regret": float(group["max_regret"].mean()),
                "max_regret_std": float(group["max_regret"].std(ddof=1)),
                "abandonment": float(group["abandonment"].mean()),
                "abandonment_std": float(group["abandonment"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lrrrrrr}", "\\toprule"]
    rows.append("Policy & Obj. & Eff. & Mean reg. & P95 reg. & Max reg. & Aband. \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        max_text = "--" if pd.isna(row["max_regret"]) else f"{row['max_regret']:.3f}"
        rows.append(
            " & ".join(
                [
                    str(row["policy"]).replace("Payment-audited queue-aware McAfee", "Audited queue McAfee").replace("Neural dynamic mechanism", "Neural"),
                    f"{row['objective']:.3f}",
                    f"{row['efficiency']:.3f}",
                    f"{row['mean_regret']:.3f}",
                    f"{row['p95_regret']:.3f}",
                    max_text,
                    f"{row['abandonment']:.3f}",
                ]
            )
            + " \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    queue = load_queue_aware()
    neural = load_neural()
    by_seed = pd.concat([neural, queue], ignore_index=True)
    summary = summarize(by_seed)
    out_dir = Path("paper/tables")
    by_seed.to_csv(out_dir / "queue_aware_certified_by_seed.csv", index=False)
    summary.to_csv(out_dir / "queue_aware_certified_audit.csv", index=False)
    write_latex_table(summary, out_dir / "queue_aware_certified_audit.tex")
    print(summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
