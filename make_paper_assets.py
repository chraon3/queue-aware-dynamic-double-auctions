from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY = Path("experiments_summary/frontier_summary.csv")
PAPER = Path("paper")
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


LABELS = {
    "dual_uniform_2x2_seed41": "Uniform 2x2",
    "ranked_dual_uniform_3x3_seed51__audit": "Uniform 3x3, ranked audit",
    "al_ranked_uniform_3x3_seed61": "Uniform 3x3, augmented",
    "ranked_dual_uniform_5x5_seed53": "Uniform 5x5, ranked",
    "dual_uniform_7x7_seed43": "Uniform 7x7, stress",
    "dual_uniform_3x3_seed7__ood_beta_easy": "Uniform-trained on beta-easy",
    "dual_uniform_3x3_seed7__ood_correlated": "Uniform-trained on correlated",
}

DYNAMIC_POLICY_LABELS = {
    "Neural dynamic mechanism": "Neural",
    "Myopic first best": "First best",
    "Dynamic McAfee": "McAfee",
    "Queue-aware McAfee": "Queue McAfee",
    "Dynamic posted price": "Posted price",
    "Optimized state posted price": "State posted",
    "Dynamic trade reduction": "Trade reduction",
    "No trade": "No trade",
}


def fmt(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Environment & Neural & McAfee & Posted price & Trade reduction & Regret \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['label']} & {fmt(row['neural_efficiency'])} & {fmt(row['mcafee_efficiency'])} & "
            f"{fmt(row['posted_price_efficiency'])} & {fmt(row['trade_reduction_efficiency'])} & "
            f"{fmt(row['total_mean_regret'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_seed_table(summary: pd.DataFrame, path: Path) -> None:
    seed_runs = [
        "dual_uniform_3x3_seed7__hybrid",
        "dual_uniform_3x3_seed21__hybrid",
        "dual_uniform_3x3_seed29__hybrid",
    ]
    df = summary[summary["run"].isin(seed_runs)].copy()
    stats = {
        "neural_efficiency": (df["neural_efficiency"].mean(), df["neural_efficiency"].std()),
        "mcafee_efficiency": (df["mcafee_efficiency"].mean(), df["mcafee_efficiency"].std()),
        "posted_price_efficiency": (df["posted_price_efficiency"].mean(), df["posted_price_efficiency"].std()),
        "total_mean_regret": (df["total_mean_regret"].mean(), df["total_mean_regret"].std()),
    }
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Mean & Std. dev. \\\\",
        "\\midrule",
    ]
    nice = {
        "neural_efficiency": "Neural efficiency",
        "mcafee_efficiency": "McAfee efficiency",
        "posted_price_efficiency": "Posted-price efficiency",
        "total_mean_regret": "Total mean regret",
    }
    for key, (mean, std) in stats.items():
        lines.append(f"{nice[key]} & {fmt(mean)} & {fmt(std)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_static_certificate_assets() -> None:
    source = Path("experiments/static_ic_certificate/static_ic_certificate_by_run.csv")
    if not source.exists():
        return
    df = pd.read_csv(source)
    df.to_csv(TABLES / "static_ic_certificate.csv", index=False)
    row = df.iloc[0]
    rows = [
        ("Audit grid size", row["grid_size"]),
        ("Grid regret, mean", row["grid_regret_mean"]),
        ("Grid regret, P95", row["grid_regret_p95"]),
        ("Lipschitz estimate, P95", row["lipschitz_p95"]),
        ("Certified regret, mean", row["certified_regret_mean"]),
        ("Certified regret, P95", row["certified_regret_p95"]),
        ("Certified regret, max", row["certified_regret_max"]),
    ]
    lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Diagnostic & Value \\\\",
        "\\midrule",
    ]
    for name, value in rows:
        if name == "Audit grid size":
            value_text = f"{int(value)}"
        else:
            value_text = fmt(value)
        lines.append(f"{name} & {value_text} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "static_ic_certificate.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_main(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    x = range(len(df))
    width = 0.22
    ax.bar([i - width for i in x], df["neural_efficiency"], width=width, label="Neural", color="#0f766e")
    ax.bar(x, df["mcafee_efficiency"], width=width, label="McAfee", color="#ea580c")
    ax.bar([i + width for i in x], df["posted_price_efficiency"], width=width, label="Posted price", color="#334155")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=25, ha="right")
    ax.set_ylabel("Efficiency relative to first best")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_efficiency.png")
    plt.close(fig)


def plot_regret(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    ax.scatter(df["total_mean_regret"], df["neural_efficiency"], s=80, color="#0f766e")
    for _, row in df.iterrows():
        ax.annotate(row["label"], (row["total_mean_regret"], row["neural_efficiency"]), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Hybrid ex-post regret")
    ax.set_ylabel("Neural efficiency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "regret_efficiency.png")
    plt.close(fig)


def write_dynamic_assets() -> None:
    run_dirs = sorted(Path("experiments").glob("dynamic_patience_value_seed*"))
    metric_paths = []
    for run_dir in run_dirs:
        recheck = run_dir / "metrics_recheck.json"
        metric_paths.append(recheck if recheck.exists() else run_dir / "metrics.json")
    if not metric_paths:
        metric_paths = [Path("experiments/dynamic_queue_ranked_seed71/metrics_dynamic_baselines.json")]
    metric_paths = [path for path in metric_paths if path.exists()]
    if not metric_paths:
        return
    raw_rows = []
    for path in metric_paths:
        metrics = pd.Series(__import__("json").loads(path.read_text(encoding="utf-8")))
        run = path.parent.name
        raw_rows.extend(
            [
                (run, "Neural dynamic mechanism", metrics["dynamic_neural_efficiency"], metrics["objective"], metrics["mean_regret"], metrics.get("p95_regret", float("nan")), metrics.get("bellman_residual", float("nan")), metrics.get("pathwise_bellman_residual", float("nan")), metrics.get("value_mae", float("nan")), metrics.get("value_rmse", float("nan")), metrics.get("mean_abandonment", float("nan"))),
                (run, "Myopic first best", 1.0, metrics["dynamic_first_best_objective"], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), metrics.get("dynamic_first_best_abandonment", float("nan"))),
                (run, "Dynamic McAfee", metrics["dynamic_mcafee_efficiency"], metrics["dynamic_mcafee_objective"], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), metrics.get("dynamic_mcafee_abandonment", float("nan"))),
                (run, "Dynamic posted price", metrics["dynamic_posted_efficiency"], metrics["dynamic_posted_objective"], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), metrics.get("dynamic_posted_abandonment", float("nan"))),
            ]
        )
        if "dynamic_state_posted_efficiency" in metrics:
            raw_rows.append((run, "Optimized state posted price", metrics["dynamic_state_posted_efficiency"], metrics["dynamic_state_posted_objective"], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), metrics.get("dynamic_state_posted_abandonment", float("nan"))))
        raw_rows.extend(
            [
                (run, "Dynamic trade reduction", metrics["dynamic_trade_reduction_efficiency"], metrics["dynamic_trade_reduction_objective"], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), metrics.get("dynamic_trade_reduction_abandonment", float("nan"))),
                (run, "No trade", metrics["dynamic_no_trade_efficiency"], metrics["dynamic_no_trade_objective"], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), metrics.get("dynamic_no_trade_abandonment", float("nan"))),
            ]
        )
        if "dynamic_queue_mcafee_efficiency" in metrics:
            raw_rows.append((run, "Queue-aware McAfee", metrics["dynamic_queue_mcafee_efficiency"], metrics["dynamic_queue_mcafee_objective"], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), metrics.get("dynamic_queue_mcafee_abandonment", float("nan"))))
    raw = pd.DataFrame(raw_rows, columns=["run", "policy", "efficiency", "objective", "regret", "p95_regret", "bellman_residual", "pathwise_bellman_residual", "value_mae", "value_rmse", "abandonment"])
    raw.to_csv(TABLES / "dynamic_results_by_seed.csv", index=False)
    df = raw.groupby("policy", sort=False).agg(
        efficiency=("efficiency", "mean"),
        efficiency_std=("efficiency", "std"),
        objective=("objective", "mean"),
        objective_std=("objective", "std"),
        regret=("regret", "mean"),
        regret_std=("regret", "std"),
        p95_regret=("p95_regret", "mean"),
        bellman_residual=("bellman_residual", "mean"),
        pathwise_bellman_residual=("pathwise_bellman_residual", "mean"),
        value_mae=("value_mae", "mean"),
        value_rmse=("value_rmse", "mean"),
        abandonment=("abandonment", "mean"),
    ).reset_index()
    policy_order = [
        "Neural dynamic mechanism",
        "Myopic first best",
        "Dynamic McAfee",
        "Dynamic posted price",
        "Optimized state posted price",
        "Dynamic trade reduction",
        "No trade",
        "Queue-aware McAfee",
    ]
    df["policy_order"] = df["policy"].map({name: idx for idx, name in enumerate(policy_order)})
    df = df.sort_values("policy_order").drop(columns=["policy_order"]).reset_index(drop=True)
    df.to_csv(TABLES / "dynamic_results.csv", index=False)
    main_df = df[df["policy"] != "Queue-aware McAfee"].copy()
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Policy & Efficiency & Objective & Regret & P95 & Aband. \\\\",
        "\\midrule",
    ]
    for _, row in main_df.iterrows():
        eff = f"{fmt(row['efficiency'])}"
        if not pd.isna(row["efficiency_std"]):
            eff += f" ({fmt(row['efficiency_std'])})"
        obj = f"{fmt(row['objective'])}"
        if not pd.isna(row["objective_std"]):
            obj += f" ({fmt(row['objective_std'])})"
        label = DYNAMIC_POLICY_LABELS.get(row["policy"], row["policy"])
        lines.append(f"{label} & {eff} & {obj} & {fmt(row['regret'])} & {fmt(row['p95_regret'])} & {fmt(row['abandonment'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "dynamic_results.tex").write_text("\n".join(lines), encoding="utf-8")

    diagnostic_df = df[df["policy"].isin(["Neural dynamic mechanism", "Optimized state posted price", "Queue-aware McAfee"])].copy()
    diagnostic_df.to_csv(TABLES / "dynamic_diagnostic_baselines.csv", index=False)
    diagnostic_lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Policy & Efficiency & Objective & Aband. \\\\",
        "\\midrule",
    ]
    for _, row in diagnostic_df.iterrows():
        diagnostic_lines.append(
            f"{DYNAMIC_POLICY_LABELS.get(row['policy'], row['policy'])} & {fmt(row['efficiency'])} & "
            f"{fmt(row['objective'])} & {fmt(row['abandonment'])} \\\\"
        )
    diagnostic_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "dynamic_diagnostic_baselines.tex").write_text("\n".join(diagnostic_lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    plot_df = main_df[main_df["policy"] != "No trade"]
    ax.bar(
        plot_df["policy"],
        plot_df["efficiency"],
        yerr=plot_df["efficiency_std"].fillna(0.0),
        capsize=3,
        color=["#0f766e", "#64748b", "#ea580c", "#2563eb", "#334155", "#7c3aed", "#991b1b"][: len(plot_df)],
    )
    ax.set_ylabel("Efficiency relative to dynamic first best")
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "dynamic_efficiency.png")
    plt.close(fig)

    neural = df[df["policy"] == "Neural dynamic mechanism"].iloc[0]
    value_lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Diagnostic & Value \\\\",
        "\\midrule",
        f"Bellman moment residual & {fmt(neural['bellman_residual'])} \\\\",
        f"Pathwise TD error & {fmt(neural['pathwise_bellman_residual'])} \\\\",
        f"Value MAE & {fmt(neural['value_mae'])} \\\\",
        f"Value RMSE & {fmt(neural['value_rmse'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    (TABLES / "value_diagnostics.tex").write_text("\n".join(value_lines), encoding="utf-8")


def write_robustness_assets() -> None:
    source = Path("experiments/dynamic_robustness/robustness_summary.csv")
    if not source.exists():
        return
    df = pd.read_csv(source)
    df.to_csv(TABLES / "dynamic_robustness.csv", index=False)
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Scenario & Neural & McAfee & Posted & Regret & P95 \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        neural = fmt(row["neural_efficiency"])
        if "neural_efficiency_std" in row and not pd.isna(row["neural_efficiency_std"]):
            neural += f" ({fmt(row['neural_efficiency_std'])})"
        lines.append(
            f"{row['label']} & {neural} & {fmt(row['mcafee_efficiency'])} & "
            f"{fmt(row['posted_efficiency'])} & {fmt(row['mean_regret'])} & {fmt(row['p95_regret'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "dynamic_robustness.tex").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=180)
    x = range(len(df))
    width = 0.24
    ax.bar([i - width for i in x], df["neural_efficiency"], width=width, label="Neural", color="#0f766e")
    ax.bar(x, df["mcafee_efficiency"], width=width, label="McAfee", color="#ea580c")
    ax.bar([i + width for i in x], df["posted_efficiency"], width=width, label="Posted", color="#334155")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=25, ha="right")
    ax.set_ylabel("Efficiency relative to dynamic first best")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "dynamic_robustness.png")
    plt.close(fig)


def write_soft_baseline_assets() -> None:
    soft_paths = sorted(Path("experiments").glob("soft_dynamic_seed*/metrics.json"))
    hard_path = TABLES / "dynamic_results.csv"
    if not soft_paths or not hard_path.exists():
        return
    hard = pd.read_csv(hard_path)
    hard_by_seed = pd.read_csv(TABLES / "dynamic_results_by_seed.csv") if (TABLES / "dynamic_results_by_seed.csv").exists() else pd.DataFrame()
    hard_neural = hard[hard["policy"] == "Neural dynamic mechanism"].iloc[0]
    soft_rows = []
    for path in soft_paths:
        metrics = pd.Series(__import__("json").loads(path.read_text(encoding="utf-8")))
        soft_rows.append(
            {
                "run": path.parent.name,
                "efficiency": metrics["dynamic_soft_efficiency"],
                "regret": metrics["mean_regret"],
                "p95_regret": metrics["p95_regret"],
                "budget_violation": metrics["budget_violation"],
                "ir_violation": metrics["total_ir_violation"],
            }
        )
    soft = pd.DataFrame(soft_rows)
    rows = [
        {
            "mechanism": "Hard-constrained",
            "seeds": int((hard_by_seed["policy"] == "Neural dynamic mechanism").sum()) if not hard_by_seed.empty else 10,
            "efficiency": hard_neural["efficiency"],
            "efficiency_std": hard_neural["efficiency_std"],
            "regret": hard_neural["regret"],
            "p95_regret": hard_neural["p95_regret"],
            "budget_violation": 0.0,
            "ir_violation": 0.0,
        },
        {
            "mechanism": "Soft-penalty",
            "seeds": len(soft),
            "efficiency": soft["efficiency"].mean(),
            "efficiency_std": soft["efficiency"].std(),
            "regret": soft["regret"].mean(),
            "p95_regret": soft["p95_regret"].mean(),
            "budget_violation": soft["budget_violation"].mean(),
            "ir_violation": soft["ir_violation"].mean(),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "soft_baseline.csv", index=False)
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Mech. & Seeds & Efficiency & Regret & P95 & Viol. \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        eff = fmt(row["efficiency"])
        if not pd.isna(row["efficiency_std"]):
            eff += f" ({fmt(row['efficiency_std'])})"
        violation = row["budget_violation"] + row["ir_violation"]
        lines.append(
            f"{'Hard' if row['mechanism'] == 'Hard-constrained' else 'Soft'} & {int(row['seeds'])} & {eff} & "
            f"{fmt(row['regret'])} & {fmt(row['p95_regret'])} & {fmt(violation)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "soft_baseline.tex").write_text("\n".join(lines), encoding="utf-8")


def write_dynamic_ablation_assets() -> None:
    hard_path = TABLES / "dynamic_results.csv"
    hard_by_seed_path = TABLES / "dynamic_results_by_seed.csv"
    ablation_path = Path("experiments/dynamic_ablations/dynamic_ablation_by_run.csv")
    soft_path = TABLES / "soft_baseline.csv"
    if not hard_path.exists() or not ablation_path.exists():
        return

    continuation_sources = {
        "Full": read_summary_row(Path("experiments/dynamic_continuation_audit/continuation_audit_summary.csv")),
        "No value loss": read_summary_row(Path("experiments/dynamic_continuation_audit_no_value/continuation_audit_summary.csv")),
        "Hybrid regret": read_summary_row(Path("experiments/dynamic_continuation_audit_hybrid_regret/continuation_audit_summary.csv")),
    }

    hard = pd.read_csv(hard_path)
    hard_by_seed = pd.read_csv(hard_by_seed_path) if hard_by_seed_path.exists() else pd.DataFrame()
    neural = hard[hard["policy"] == "Neural dynamic mechanism"].iloc[0]
    rows = [
        {
            "mechanism": "Full",
            "seeds": int((hard_by_seed["policy"] == "Neural dynamic mechanism").sum()) if not hard_by_seed.empty else 10,
            "efficiency": neural["efficiency"],
            "efficiency_std": neural["efficiency_std"],
            "regret": neural["regret"],
            "p95_regret": neural["p95_regret"],
            "bellman_residual": neural["bellman_residual"],
            "continuation_regret": continuation_sources["Full"]["continuation_mean_regret"] if continuation_sources["Full"] is not None else float("nan"),
            "violation": 0.0,
        }
    ]

    ablations = pd.read_csv(ablation_path)
    labels = {
        "no_value": "No value loss",
        "hybrid_regret": "Hybrid regret",
    }
    for kind, label in labels.items():
        df = ablations[ablations["kind"] == kind]
        if df.empty:
            continue
        rows.append(
            {
                "mechanism": label,
                "seeds": len(df),
                "efficiency": df["dynamic_neural_efficiency"].mean(),
                "efficiency_std": df["dynamic_neural_efficiency"].std(),
                "regret": df["mean_regret"].mean(),
                "p95_regret": df["p95_regret"].mean(),
                "bellman_residual": df["bellman_residual"].mean(),
                "continuation_regret": continuation_sources[label]["continuation_mean_regret"] if continuation_sources[label] is not None else float("nan"),
                "violation": 0.0,
            }
        )

    if soft_path.exists():
        soft = pd.read_csv(soft_path)
        soft_row = soft[soft["mechanism"] == "Soft-penalty"].iloc[0]
        rows.append(
            {
                "mechanism": "Soft",
                "seeds": int(soft_row["seeds"]),
                "efficiency": soft_row["efficiency"],
                "efficiency_std": soft_row["efficiency_std"],
                "regret": soft_row["regret"],
                "p95_regret": soft_row["p95_regret"],
                "bellman_residual": float("nan"),
                "continuation_regret": float("nan"),
                "violation": soft_row["budget_violation"] + soft_row["ir_violation"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "dynamic_ablation.csv", index=False)
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Mech. & Seeds & Eff. & Regret & P95 & Bell. & Cont. & Viol. \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        eff = fmt(row["efficiency"])
        if not pd.isna(row["efficiency_std"]):
            eff += f" ({fmt(row['efficiency_std'])})"
        lines.append(
            f"{row['mechanism']} & {int(row['seeds'])} & {eff} & {fmt(row['regret'])} & "
            f"{fmt(row['p95_regret'])} & {fmt(row['bellman_residual'])} & "
            f"{fmt(row['continuation_regret'])} & {fmt(row['violation'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "dynamic_ablation.tex").write_text("\n".join(lines), encoding="utf-8")


def write_history_audit_assets() -> None:
    source = Path("experiments/dynamic_history_audit/history_audit_summary.csv")
    if not source.exists():
        return
    summary = pd.read_csv(source)
    summary.to_csv(TABLES / "history_audit.csv", index=False)
    mean = summary[summary["stat"] == "mean"].iloc[0]
    std = summary[summary["stat"] == "std"].iloc[0]
    rows = [
        ("Episodes per seed", mean["episodes"], std["episodes"]),
        ("Candidate strategies", mean["candidate_count"], std["candidate_count"]),
        ("History Sobol draws", mean["history_draws"], std["history_draws"]),
        ("History audit mean regret", mean["history_mean_regret"], std["history_mean_regret"]),
        ("History audit P95 regret", mean["history_p95_regret"], std["history_p95_regret"]),
        ("History audit max regret", mean["history_max_regret"], std["history_max_regret"]),
    ]
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Mean & Std. dev. \\\\",
        "\\midrule",
    ]
    for name, value, value_std in rows:
        if name in {"Episodes per seed", "Candidate strategies", "History Sobol draws"}:
            value_text = f"{int(round(value))}"
            std_text = "--" if pd.isna(value_std) or abs(value_std) < 1.0e-12 else f"{int(round(value_std))}"
        else:
            value_text = fmt(value)
            std_text = fmt(value_std)
        lines.append(f"{name} & {value_text} & {std_text} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "history_audit.tex").write_text("\n".join(lines), encoding="utf-8")


def write_continuation_audit_assets() -> None:
    source = Path("experiments/dynamic_continuation_audit/continuation_audit_summary.csv")
    if not source.exists():
        return
    summary = pd.read_csv(source)
    summary.to_csv(TABLES / "continuation_audit.csv", index=False)
    mean = summary[summary["stat"] == "mean"].iloc[0]
    std = summary[summary["stat"] == "std"].iloc[0]
    rows = [
        ("Episodes per side/seed", mean["episodes_per_side"], std["episodes_per_side"]),
        ("Report rules", mean["report_strategy_count"], std["report_strategy_count"]),
        ("Exit thresholds", mean["exit_threshold_count"], std["exit_threshold_count"]),
        ("Continuation strategies", mean["continuation_strategy_count"], std["continuation_strategy_count"]),
        ("No-exit mean regret", mean["no_exit_continuation_mean_regret"], std["no_exit_continuation_mean_regret"]),
        ("No-exit P95 regret", mean["no_exit_continuation_p95_regret"], std["no_exit_continuation_p95_regret"]),
        ("Exit mean regret", mean["continuation_mean_regret"], std["continuation_mean_regret"]),
        ("Exit P95 regret", mean["continuation_p95_regret"], std["continuation_p95_regret"]),
        ("Exit max regret", mean["continuation_max_regret"], std["continuation_max_regret"]),
    ]
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Mean & Std. dev. \\\\",
        "\\midrule",
    ]
    count_rows = {"Episodes per side/seed", "Report rules", "Exit thresholds", "Continuation strategies"}
    for name, value, value_std in rows:
        if name in count_rows:
            value_text = f"{int(round(value))}"
            std_text = "--" if pd.isna(value_std) or abs(value_std) < 1.0e-12 else f"{int(round(value_std))}"
        else:
            value_text = fmt(value)
            std_text = fmt(value_std)
        lines.append(f"{name} & {value_text} & {std_text} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "continuation_audit.tex").write_text("\n".join(lines), encoding="utf-8")


def read_summary_row(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    rows = summary[summary["stat"] == "mean"]
    if rows.empty:
        return None
    return rows.iloc[0]


def write_continuation_training_assets() -> None:
    full_cont = read_summary_row(Path("experiments/dynamic_continuation_audit/continuation_audit_summary.csv"))
    tuned_cont = read_summary_row(Path("experiments/dynamic_continuation_audit_finetuned/continuation_audit_summary.csv"))
    tuned_hist = read_summary_row(Path("experiments/dynamic_history_audit_finetuned/history_audit_summary.csv"))
    tail_cont = read_summary_row(Path("experiments/dynamic_continuation_audit_tail/continuation_audit_summary.csv"))
    tail_hist = read_summary_row(Path("experiments/dynamic_history_audit_tail/history_audit_summary.csv"))
    hard_tail_cont = read_summary_row(Path("experiments/dynamic_continuation_audit_hardcase_select/continuation_audit_summary.csv"))
    hard_tail_hist = read_summary_row(Path("experiments/dynamic_history_audit_hardcase_select/history_audit_summary.csv"))
    full_hist = read_summary_row(Path("experiments/dynamic_history_audit/history_audit_summary.csv"))
    hard_path = TABLES / "dynamic_results.csv"
    if full_cont is None or not hard_path.exists():
        return
    hard = pd.read_csv(hard_path)
    neural = hard[hard["policy"] == "Neural dynamic mechanism"].iloc[0]

    def collect_metrics(pattern: str) -> pd.DataFrame:
        rows = []
        for path in sorted(Path("experiments").glob(pattern)):
            metric_path = path.with_name("metrics_recheck.json") if path.name == "metrics.json" else path
            if not metric_path.exists():
                metric_path = path
            metrics = pd.Series(__import__("json").loads(metric_path.read_text(encoding="utf-8")))
            rows.append(metrics)
        return pd.DataFrame(rows)

    tuned = collect_metrics("continuation_finetune_seed*/metrics.json")
    tail = collect_metrics("continuation_tail_seed*/metrics.json")
    hard_tail = collect_metrics("continuation_hardcase_select_seed*/metrics.json")
    rows = [
        {
            "mechanism": "Full",
            "seeds": 10,
            "efficiency": neural["efficiency"],
            "regret": neural["regret"],
            "p95_regret": neural["p95_regret"],
            "history_regret": full_hist["history_mean_regret"] if full_hist is not None else float("nan"),
            "continuation_regret": full_cont["continuation_mean_regret"],
            "continuation_p95": full_cont["continuation_p95_regret"],
            "continuation_max": full_cont["continuation_max_regret"],
        }
    ]
    if not tuned.empty and tuned_cont is not None:
        rows.append(
            {
                "mechanism": "Mean",
                "seeds": len(tuned),
                "efficiency": tuned["dynamic_neural_efficiency"].mean(),
                "regret": tuned["mean_regret"].mean(),
                "p95_regret": tuned["p95_regret"].mean(),
                "history_regret": tuned_hist["history_mean_regret"] if tuned_hist is not None else float("nan"),
                "continuation_regret": tuned_cont["continuation_mean_regret"],
                "continuation_p95": tuned_cont["continuation_p95_regret"],
                "continuation_max": tuned_cont["continuation_max_regret"],
            }
        )
    if not tail.empty and tail_cont is not None:
        rows.append(
            {
                "mechanism": "Tail",
                "seeds": len(tail),
                "efficiency": tail["dynamic_neural_efficiency"].mean(),
                "regret": tail["mean_regret"].mean(),
                "p95_regret": tail["p95_regret"].mean(),
                "history_regret": tail_hist["history_mean_regret"] if tail_hist is not None else float("nan"),
                "continuation_regret": tail_cont["continuation_mean_regret"],
                "continuation_p95": tail_cont["continuation_p95_regret"],
                "continuation_max": tail_cont["continuation_max_regret"],
            }
        )
    if not hard_tail.empty and hard_tail_cont is not None:
        rows.append(
            {
                "mechanism": "Hard",
                "seeds": len(hard_tail),
                "efficiency": hard_tail["dynamic_neural_efficiency"].mean(),
                "regret": hard_tail["mean_regret"].mean(),
                "p95_regret": hard_tail["p95_regret"].mean(),
                "history_regret": hard_tail_hist["history_mean_regret"] if hard_tail_hist is not None else float("nan"),
                "continuation_regret": hard_tail_cont["continuation_mean_regret"],
                "continuation_p95": hard_tail_cont["continuation_p95_regret"],
                "continuation_max": hard_tail_cont["continuation_max_regret"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "continuation_training.csv", index=False)
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Mech. & Seeds & Eff. & Reg. & Hist. & Cont. & P95 & Max \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['mechanism']} & {int(row['seeds'])} & {fmt(row['efficiency'])} & "
            f"{fmt(row['regret'])} & {fmt(row['history_regret'])} & "
            f"{fmt(row['continuation_regret'])} & {fmt(row['continuation_p95'])} & "
            f"{fmt(row['continuation_max'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "continuation_training.tex").write_text("\n".join(lines), encoding="utf-8")


def write_continuation_worstcase_assets() -> None:
    tail_path = Path("experiments/dynamic_continuation_audit_tail_details/continuation_audit_by_run.csv")
    hard_path = Path("experiments/dynamic_continuation_audit_hardcase_select_details/continuation_audit_by_run.csv")
    pilot_path = Path("experiments/dynamic_continuation_audit_auditmatched_pilot_seed91/continuation_audit_by_run.csv")
    rows = []
    exit_rows = []
    for label, detail_path in [
        ("Tail", Path("experiments/dynamic_continuation_audit_tail_details/continuation_audit_worst_cases.csv")),
        ("Hard", Path("experiments/dynamic_continuation_audit_hardcase_select_details/continuation_audit_worst_cases.csv")),
    ]:
        if detail_path.exists():
            details = pd.read_csv(detail_path)
            if not details.empty:
                counts = details.groupby("exit_threshold").size().to_dict()
                exit_rows.append(
                    {
                        "variant": label,
                        "saved_cases": len(details),
                        "exit_2": int(counts.get(2.0, 0)),
                        "exit_4": int(counts.get(4.0, 0)),
                        "exit_99": int(counts.get(99.0, 0)),
                    }
                )
    if tail_path.exists() and hard_path.exists():
        tail = pd.read_csv(tail_path).set_index("run")
        hard = pd.read_csv(hard_path).set_index("run")
        hard.index = hard.index.str.replace("continuation_hardcase_select_", "continuation_tail_", regex=False)
        common = tail.index.intersection(hard.index)
        if len(common) > 0:
            comparison = pd.DataFrame(
                {
                    "tail_p95": tail.loc[common, "continuation_p95_regret"],
                    "hard_p95": hard.loc[common, "continuation_p95_regret"],
                    "tail_max": tail.loc[common, "continuation_max_regret"],
                    "hard_max": hard.loc[common, "continuation_max_regret"],
                }
            )
            comparison["delta_p95"] = comparison["hard_p95"] - comparison["tail_p95"]
            comparison["delta_max"] = comparison["hard_max"] - comparison["tail_max"]
            rows.append(
                {
                    "diagnostic": "Hard vs Tail",
                    "scope": f"{len(common)} seeds",
                    "delta_p95": comparison["delta_p95"].mean(),
                    "delta_max": comparison["delta_max"].mean(),
                    "p95_wins": f"{int((comparison['delta_p95'] < 0).sum())}/{len(common)}",
                    "max_wins": f"{int((comparison['delta_max'] < 0).sum())}/{len(common)}",
                }
            )
            comparison.to_csv(TABLES / "continuation_hard_vs_tail_by_seed.csv")
    if pilot_path.exists():
        pilot = pd.read_csv(pilot_path).set_index("run")
        base_name = "continuation_hardcase_select_seed91"
        pilot_name = "continuation_auditmatched_seed91"
        if base_name in pilot.index and pilot_name in pilot.index:
            delta_p95 = pilot.loc[pilot_name, "continuation_p95_regret"] - pilot.loc[base_name, "continuation_p95_regret"]
            delta_max = pilot.loc[pilot_name, "continuation_max_regret"] - pilot.loc[base_name, "continuation_max_regret"]
            rows.append(
                {
                    "diagnostic": "Pilot",
                    "scope": "seed 91",
                    "delta_p95": delta_p95,
                    "delta_max": delta_max,
                    "p95_wins": "0/1" if delta_p95 >= 0 else "1/1",
                    "max_wins": "1/1" if delta_max < 0 else "0/1",
                }
            )
    if exit_rows:
        pd.DataFrame(exit_rows).to_csv(TABLES / "continuation_worstcase_exit_counts.csv", index=False)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "continuation_worstcase_diagnostics.csv", index=False)
    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Diagnostic & Scope & dP95 & dMax & P95 wins & Max wins \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['diagnostic']} & {row['scope']} & {fmt(row['delta_p95'])} & "
            f"{fmt(row['delta_max'])} & {row['p95_wins']} & {row['max_wins']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "continuation_worstcase_diagnostics.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    selected = summary[summary["run"].isin(LABELS)].copy()
    selected["label"] = selected["run"].map(LABELS)
    selected["order"] = selected["run"].map({run: i for i, run in enumerate(LABELS)})
    selected = selected.sort_values("order")
    selected.to_csv(TABLES / "main_results.csv", index=False)
    write_latex_table(selected, TABLES / "main_results.tex")
    write_seed_table(summary, TABLES / "uniform_3x3_seed_stats.tex")
    write_static_certificate_assets()
    plot_main(selected)
    plot_regret(selected)
    write_dynamic_assets()
    write_robustness_assets()
    write_soft_baseline_assets()
    write_dynamic_ablation_assets()
    write_history_audit_assets()
    write_continuation_audit_assets()
    write_continuation_training_assets()
    write_continuation_worstcase_assets()

if __name__ == "__main__":
    main()
