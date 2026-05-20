from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCES = [
    (
        "One-period",
        "Affine history",
        "experiments/dynamic_history_audit/history_audit_by_run.csv",
        {
            "rules": "candidate_count",
            "mean": "history_mean_regret",
            "p95": "history_p95_regret",
            "max": "history_max_regret",
        },
    ),
    (
        "One-period",
        "Nonlinear history",
        "experiments/dynamic_history_audit_history_nonlinear/history_audit_by_run.csv",
        {
            "rules": "candidate_count",
            "mean": "history_mean_regret",
            "p95": "history_p95_regret",
            "max": "history_max_regret",
        },
    ),
    (
        "One-period",
        "Piecewise history",
        "experiments/dynamic_history_audit_history_piecewise/history_audit_by_run.csv",
        {
            "rules": "candidate_count",
            "mean": "history_mean_regret",
            "p95": "history_p95_regret",
            "max": "history_max_regret",
        },
    ),
    (
        "Continuation",
        "Affine history",
        "experiments/dynamic_continuation_audit/continuation_audit_by_run.csv",
        {
            "rules": "continuation_strategy_count",
            "mean": "continuation_mean_regret",
            "p95": "continuation_p95_regret",
            "max": "continuation_max_regret",
        },
    ),
    (
        "Continuation",
        "Nonlinear history",
        "experiments/dynamic_continuation_audit_history_nonlinear/continuation_audit_by_run.csv",
        {
            "rules": "continuation_strategy_count",
            "mean": "continuation_mean_regret",
            "p95": "continuation_p95_regret",
            "max": "continuation_max_regret",
        },
    ),
    (
        "Continuation",
        "Piecewise history",
        "experiments/dynamic_continuation_audit_history_piecewise/continuation_audit_by_run.csv",
        {
            "rules": "continuation_strategy_count",
            "mean": "continuation_mean_regret",
            "p95": "continuation_p95_regret",
            "max": "continuation_max_regret",
        },
    ),
]


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{llrrrr}", "\\toprule"]
    rows.append("Audit & Family & Rules & Mean & P95 & Max \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["audit"]),
                    str(row["family"]),
                    f"{row['rules']:.0f}",
                    f"{row['mean']:.3f}",
                    f"{row['p95']:.3f}",
                    f"{row['max']:.3f}",
                ]
            )
            + " \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    rows = []
    for audit, family, source, columns in SOURCES:
        data = pd.read_csv(source)
        rows.append(
            {
                "audit": audit,
                "family": family,
                "runs": len(data),
                "rules": float(data[columns["rules"]].mean()),
                "mean": float(data[columns["mean"]].mean()),
                "mean_std": float(data[columns["mean"]].std(ddof=1)),
                "p95": float(data[columns["p95"]].mean()),
                "p95_std": float(data[columns["p95"]].std(ddof=1)),
                "max": float(data[columns["max"]].mean()),
                "max_std": float(data[columns["max"]].std(ddof=1)),
            }
        )
    summary = pd.DataFrame(rows)
    out_csv = Path("paper/tables/nonlinear_deviation_audit.csv")
    out_tex = Path("paper/tables/nonlinear_deviation_audit.tex")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    write_latex_table(summary, out_tex)
    print(summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
