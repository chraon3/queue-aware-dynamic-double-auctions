from __future__ import annotations

import argparse
import csv
import re
import struct
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
DEFAULT_OUT = PAPER / "submission_package" / "figure_audit_report.md"


@dataclass
class FigureFinding:
    figure: str
    level: str
    message: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_includegraphics(text: str) -> list[tuple[int, str]]:
    out = []
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", text):
        line = text.count("\n", 0, match.start()) + 1
        out.append((line, match.group(1)))
    return out


def png_dimensions(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def audit_included_figures(main_tex: Path) -> list[FigureFinding]:
    text = read_text(main_tex)
    findings: list[FigureFinding] = []
    for line, raw in parse_includegraphics(text):
        path = main_tex.parent / raw
        label = f"line {line}: {raw}"
        if not path.exists():
            findings.append(FigureFinding(label, "error", "Referenced figure file is missing."))
            continue
        if path.suffix.lower() != ".pdf":
            findings.append(FigureFinding(label, "warning", "Main-text figures should normally use vector PDF for journal submission."))
        if path.stat().st_size < 2_000:
            findings.append(FigureFinding(label, "warning", f"File is unusually small ({path.stat().st_size} bytes)."))
        if path.suffix.lower() == ".pdf":
            with path.open("rb") as handle:
                header = handle.read(4)
            if header != b"%PDF":
                findings.append(FigureFinding(label, "error", "PDF header is invalid."))
    return findings


def audit_png_assets(fig_dir: Path) -> list[FigureFinding]:
    findings: list[FigureFinding] = []
    manifest = fig_dir / "redraw_manifest.csv"
    allowed_pngs: set[str] | None = None
    if manifest.exists():
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            allowed_pngs = {Path(row["png"]).name for row in reader if row.get("png")}
    for png in sorted(fig_dir.glob("*.png")):
        if allowed_pngs is not None and png.name not in allowed_pngs:
            findings.append(FigureFinding(png.name, "warning", "PNG is not listed in redraw_manifest.csv and is not part of the clean manuscript figure set."))
            continue
        dims = png_dimensions(png)
        if dims is None:
            findings.append(FigureFinding(png.name, "error", "PNG header could not be read."))
            continue
        width, height = dims
        if width < 900 or height < 550:
            findings.append(FigureFinding(png.name, "warning", f"Low raster size for fallback PNG: {width}x{height}."))
        else:
            findings.append(FigureFinding(png.name, "note", f"Fallback PNG size: {width}x{height}."))
    return findings


def audit_manifest(fig_dir: Path) -> list[FigureFinding]:
    manifest = fig_dir / "redraw_manifest.csv"
    findings: list[FigureFinding] = []
    if not manifest.exists():
        return [FigureFinding("redraw_manifest.csv", "warning", "No redraw manifest found.")]
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    expected = {"figure", "png", "pdf", "svg", "png_bytes"}
    missing_columns = expected.difference(reader.fieldnames or [])
    if missing_columns:
        findings.append(FigureFinding("redraw_manifest.csv", "error", "Missing columns: " + ", ".join(sorted(missing_columns))))
        return findings
    for row in rows:
        stem = row["figure"]
        for key in ["png", "pdf", "svg"]:
            raw = row[key]
            path = ROOT / raw if not Path(raw).is_absolute() else Path(raw)
            if not path.exists():
                fallback = fig_dir / ("vector" if key in {"pdf", "svg"} else "") / f"{stem}.{key}"
                if key == "png":
                    fallback = fig_dir / f"{stem}.png"
                if fallback.exists():
                    findings.append(FigureFinding(stem, "warning", f"Manifest path for {key} is stale or absolute but fallback exists."))
                else:
                    findings.append(FigureFinding(stem, "error", f"Manifest path for {key} is missing: {raw}"))
    if not findings:
        findings.append(FigureFinding("redraw_manifest.csv", "note", f"Manifest entries verified: {len(rows)}."))
    return findings


def write_report(out: Path, findings: list[FigureFinding]) -> None:
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    notes = [f for f in findings if f.level == "note"]
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")

    lines = [
        "# Figure Audit",
        "",
        f"Status: **{status}**",
        "",
        "The audit verifies that manuscript figures exist, use vector PDF in the main text, "
        "and have high-resolution PNG fallbacks for previews or ancillary material.",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- `{f.figure}`: {f.message}" for f in errors] or ["- None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- `{f.figure}`: {f.message}" for f in warnings] or ["- None."])
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- `{f.figure}`: {f.message}" for f in notes] or ["- None."])
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit manuscript figure assets.")
    parser.add_argument("--main-tex", type=Path, default=PAPER / "main.tex")
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    findings = []
    findings.extend(audit_included_figures(args.main_tex))
    findings.extend(audit_png_assets(args.figure_dir))
    findings.extend(audit_manifest(args.figure_dir))
    write_report(args.out, findings)

    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    print(f"Wrote {args.out}")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
