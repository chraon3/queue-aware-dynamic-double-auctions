from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "paper" / "tables"
DEFAULT_OUT = ROOT / "paper" / "submission_package" / "data_schema_audit_report.md"


SCHEMAS = {
    "main_results.csv": {
        "required": [
            "run",
            "regret_method",
            "neural_efficiency",
            "mcafee_efficiency",
            "posted_price_efficiency",
            "total_mean_regret",
            "min_budget_surplus",
            "max_clearing_abs",
            "label",
        ],
        "nonnegative": ["total_mean_regret", "mean_trade_volume", "max_clearing_abs"],
        "bounded_0_1": ["neural_efficiency", "mcafee_efficiency", "posted_price_efficiency"],
    },
    "dynamic_results.csv": {
        "required": [
            "policy",
            "efficiency",
            "objective",
            "regret",
            "p95_regret",
            "bellman_residual",
            "pathwise_bellman_residual",
            "abandonment",
        ],
        "nonnegative": ["regret", "p95_regret", "bellman_residual", "pathwise_bellman_residual", "abandonment"],
        "bounded_0_1": ["abandonment"],
    },
    "queue_aware_certified_audit.csv": {
        "required": [
            "policy",
            "runs",
            "objective",
            "efficiency",
            "mean_regret",
            "p95_regret",
            "max_regret",
            "abandonment",
        ],
        "nonnegative": ["runs", "mean_regret", "p95_regret", "max_regret", "abandonment"],
        "bounded_0_1": ["abandonment"],
    },
    "queue_aware_stress.csv": {
        "required": [
            "scenario",
            "label",
            "n_seeds",
            "neural_objective",
            "queue_objective",
            "gap_objective",
            "neural_abandonment",
            "queue_abandonment",
            "gap_abandonment",
            "queue_objective_wins",
            "queue_lower_abandonment_wins",
        ],
        "nonnegative": ["n_seeds", "neural_abandonment", "queue_abandonment"],
        "bounded_0_1": ["neural_abandonment", "queue_abandonment"],
    },
    "dynamic_robustness.csv": {
        "required": [
            "scenario",
            "label",
            "neural_efficiency",
            "mcafee_efficiency",
            "posted_efficiency",
            "mean_regret",
            "p95_regret",
            "abandonment",
        ],
        "nonnegative": ["mean_regret", "p95_regret", "abandonment"],
        "bounded_0_1": ["abandonment"],
    },
}


@dataclass
class Finding:
    file: str
    level: str
    message: str


def parse_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def as_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def audit_file(name: str, spec: dict[str, list[str]]) -> list[Finding]:
    path = TABLES / name
    findings: list[Finding] = []
    if not path.exists():
        return [Finding(name, "error", "Expected CSV file is missing.")]

    rows, columns = parse_rows(path)
    if not rows:
        findings.append(Finding(name, "error", "CSV has no data rows."))
        return findings

    missing = [col for col in spec["required"] if col not in columns]
    if missing:
        findings.append(Finding(name, "error", "Missing required columns: " + ", ".join(missing)))
        return findings

    numeric_columns = sorted(
        {
            col
            for col in columns
            if col not in {"run", "regret_method", "policy", "scenario", "label", "environment"}
        }
    )
    invalid = []
    for col in numeric_columns:
        for row_id, row in enumerate(rows, start=2):
            value = row.get(col, "")
            if value == "":
                continue
            if as_float(value) is None:
                invalid.append(f"{col} row {row_id}")
                break
    if invalid:
        findings.append(Finding(name, "error", "Non-numeric values in numeric columns: " + ", ".join(invalid[:8])))

    for col in spec.get("nonnegative", []):
        if col not in columns:
            continue
        bad = []
        for row_id, row in enumerate(rows, start=2):
            value = as_float(row.get(col, ""))
            if value is not None and value < -1e-10:
                bad.append(row_id)
        if bad:
            findings.append(Finding(name, "error", f"`{col}` has negative value(s) at row(s): {bad[:8]}"))

    for col in spec.get("bounded_0_1", []):
        if col not in columns:
            continue
        bad = []
        for row_id, row in enumerate(rows, start=2):
            value = as_float(row.get(col, ""))
            if value is not None and not (-1e-10 <= value <= 1.2):
                bad.append(row_id)
        if bad:
            findings.append(Finding(name, "warning", f"`{col}` falls outside the loose [0, 1.2] range at row(s): {bad[:8]}"))

    for col in columns:
        if col.endswith("_std"):
            bad = []
            for row_id, row in enumerate(rows, start=2):
                value = as_float(row.get(col, ""))
                if value is not None and value < -1e-10:
                    bad.append(row_id)
            if bad:
                findings.append(Finding(name, "error", f"`{col}` has negative standard deviation at row(s): {bad[:8]}"))

    findings.append(Finding(name, "note", f"Rows: {len(rows)}; columns: {len(columns)}."))
    return findings


def write_report(out: Path, findings: list[Finding]) -> None:
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    notes = [f for f in findings if f.level == "note"]
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")

    lines = [
        "# Data Schema Audit",
        "",
        f"Status: **{status}**",
        "",
        "This audit checks the submission-facing CSV tables that feed the main manuscript. "
        "It is intentionally lightweight and dependency-free, so it can be run on a fresh replication machine.",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- `{f.file}`: {f.message}" for f in errors] or ["- None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- `{f.file}`: {f.message}" for f in warnings] or ["- None."])
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- `{f.file}`: {f.message}" for f in notes] or ["- None."])
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit key paper CSV schemas.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    findings: list[Finding] = []
    for name, spec in SCHEMAS.items():
        findings.extend(audit_file(name, spec))
    write_report(args.out, findings)

    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    print(f"Wrote {args.out}")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
