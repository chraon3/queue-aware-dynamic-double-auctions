from __future__ import annotations

from pathlib import Path

import pandas as pd


FINITE_SOURCE = Path("paper/tables/nonlinear_deviation_audit.csv")
LEARNED_SOURCE = Path("experiments/learned_continuation_auditor_stronger/learned_auditor_by_run.csv")
LEARNED_REPORT_EXIT_SOURCE = Path("experiments/learned_report_exit_auditor/learned_report_exit_by_run.csv")


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule"]
    rows.append("Auditor & Runs & Rules & Mean & P95 & Max \\\\")
    rows.append("\\midrule")
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["auditor"]),
                    f"{row['runs']:.0f}",
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
    finite = pd.read_csv(FINITE_SOURCE)
    continuation = finite.loc[finite["audit"] == "Continuation"].copy()
    rows = []
    for _, row in continuation.iterrows():
        rows.append(
            {
                "auditor": row["family"],
                "runs": row["runs"],
                "rules": row["rules"],
                "mean": row["mean"],
                "p95": row["p95"],
                "max": row["max"],
                "mean_std": row["mean_std"],
                "p95_std": row["p95_std"],
                "max_std": row["max_std"],
            }
        )

    learned = pd.read_csv(LEARNED_SOURCE)
    rows.append(
        {
            "auditor": "Learned neural report",
            "runs": len(learned),
            "rules": learned["learned_strategy_count_per_side"].mean(),
            "mean": learned["learned_test_mean_regret"].mean(),
            "p95": learned["learned_test_p95_regret"].mean(),
            "max": learned["learned_test_max_regret"].mean(),
            "mean_std": learned["learned_test_mean_regret"].std(ddof=1),
            "p95_std": learned["learned_test_p95_regret"].std(ddof=1),
            "max_std": learned["learned_test_max_regret"].std(ddof=1),
        }
    )
    if LEARNED_REPORT_EXIT_SOURCE.exists():
        learned_exit = pd.read_csv(LEARNED_REPORT_EXIT_SOURCE)
        rows.append(
            {
                "auditor": "Learned report-exit",
                "runs": len(learned_exit),
                "rules": learned_exit["learned_strategy_count_per_side"].mean(),
                "mean": learned_exit["learned_report_exit_test_mean_regret"].mean(),
                "p95": learned_exit["learned_report_exit_test_p95_regret"].mean(),
                "max": learned_exit["learned_report_exit_test_max_regret"].mean(),
                "mean_std": learned_exit["learned_report_exit_test_mean_regret"].std(ddof=1),
                "p95_std": learned_exit["learned_report_exit_test_p95_regret"].std(ddof=1),
                "max_std": learned_exit["learned_report_exit_test_max_regret"].std(ddof=1),
            }
        )

    summary = pd.DataFrame(rows)
    out_csv = Path("paper/tables/learned_auditor_audit.csv")
    out_tex = Path("paper/tables/learned_auditor_audit.tex")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    write_latex_table(summary, out_tex)
    print(summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
