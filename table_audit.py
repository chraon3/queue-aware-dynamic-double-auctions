from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
TABLE_DIR = PAPER / "tables"
DEFAULT_OUT = PAPER / "submission_package" / "table_audit_report.md"


@dataclass
class TableFinding:
    table: str
    level: str
    message: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_tabular_spec(text: str) -> str | None:
    marker = r"\begin{tabular}"
    start = text.find(marker)
    if start < 0:
        return None
    brace = text.find("{", start + len(marker))
    if brace < 0:
        return None
    depth = 0
    chars: list[str] = []
    for pos in range(brace, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
        chars.append(char)
    return None


def skip_braced(spec: str, index: int) -> int:
    if index >= len(spec) or spec[index] != "{":
        return index
    depth = 0
    for pos in range(index, len(spec)):
        if spec[pos] == "{":
            depth += 1
        elif spec[pos] == "}":
            depth -= 1
            if depth == 0:
                return pos + 1
    return len(spec)


def count_columns(spec: str | None) -> int | None:
    if spec is None:
        return None
    count = 0
    idx = 0
    while idx < len(spec):
        char = spec[idx]
        if char in "lcrX":
            count += 1
            idx += 1
        elif char in "pmbL":
            count += 1
            idx = skip_braced(spec, idx + 1)
        elif char in "@><":
            idx = skip_braced(spec, idx + 1)
        else:
            idx += 1
    return count


def count_body_rows(text: str) -> int:
    rows = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or "\\" not in line:
            continue
        if any(rule in line for rule in [r"\toprule", r"\midrule", r"\bottomrule", r"\cmidrule"]):
            continue
        if "&" in line and r"\\" in line:
            rows += 1
    return rows


def referenced_tables(main_tex: Path) -> tuple[set[str], set[str], set[str]]:
    text = read_text(main_tex)
    inputs = set()
    compact = set()
    resizebox = set()
    for match in re.finditer(r"\\input\{tables/([^{}]+?)(?:\.tex)?\}", text):
        inputs.add(match.group(1) + ".tex")
    for match in re.finditer(
        r"\\inputs(?:mall|cript)table(?:\[[^\]]*\])?\{tables/([^{}]+?)(?:\.tex)?\}", text
    ):
        name = match.group(1) + ".tex"
        inputs.add(name)
        compact.add(name)
    for match in re.finditer(r"\\resizebox\{[^{}]+\}\{[^{}]+\}\{\\input\{tables/([^{}]+?)(?:\.tex)?\}\}", text):
        resizebox.add(match.group(1) + ".tex")
    return inputs, resizebox, compact


def audit_table(path: Path, referenced: bool, resized: bool, compact: bool) -> list[TableFinding]:
    text = read_text(path)
    findings: list[TableFinding] = []
    label = path.name

    if r"\begin{tabular}" not in text:
        findings.append(TableFinding(label, "error", "No tabular environment found."))
        return findings
    for rule in [r"\toprule", r"\midrule", r"\bottomrule"]:
        if rule not in text:
            findings.append(TableFinding(label, "warning", f"Missing `{rule}`; use booktabs-style tables."))

    spec = extract_tabular_spec(text)
    col_count = count_columns(spec)
    row_count = count_body_rows(text)
    if col_count is None:
        findings.append(TableFinding(label, "warning", "Could not parse column specification."))
    elif col_count > 7 and referenced and not compact:
        findings.append(TableFinding(label, "warning", f"Wide table has {col_count} columns."))
    elif col_count is not None and col_count > 7 and compact:
        findings.append(TableFinding(label, "note", f"Wide table has {col_count} columns and is handled by a compact table macro."))
    if row_count > 12 and referenced:
        findings.append(TableFinding(label, "warning", f"Long main-text table has {row_count} body rows."))
    if resized:
        findings.append(TableFinding(label, "warning", "Main manuscript wraps this table in `resizebox`; review readability before submission."))
    if not referenced:
        findings.append(TableFinding(label, "note", "Not referenced by `paper/main.tex`; keep only if appendix/provenance needs it."))
    return findings


def write_report(out: Path, findings: list[TableFinding], total_tables: int, referenced_count: int) -> None:
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    notes = [f for f in findings if f.level == "note"]
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")

    lines = [
        "# Table Audit",
        "",
        f"Status: **{status}**",
        f"Tables scanned: {total_tables}",
        f"Tables referenced by `paper/main.tex`: {referenced_count}",
        "",
        "The audit emphasizes submission layout risks rather than numerical interpretation. "
        "Wide or resized tables should either be shortened, moved to the appendix, or redesigned before journal submission.",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- `{f.table}`: {f.message}" for f in errors] or ["- None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- `{f.table}`: {f.message}" for f in warnings] or ["- None."])
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- `{f.table}`: {f.message}" for f in notes] or ["- None."])
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit paper table formatting and layout risks.")
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--main-tex", type=Path, default=PAPER / "main.tex")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    referenced, resized, compact = referenced_tables(args.main_tex)
    tables = sorted(args.table_dir.glob("*.tex"))
    findings: list[TableFinding] = []
    for table in tables:
        findings.extend(audit_table(table, table.name in referenced, table.name in resized, table.name in compact))
    write_report(args.out, findings, len(tables), len(referenced))

    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    print(f"Wrote {args.out}")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
