from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = ROOT / "paper" / "build" / "main.pdf"
DEFAULT_OUT_DIR = ROOT / "paper" / "build" / "visual_audit"
DEFAULT_REPORT = ROOT / "paper" / "submission_package" / "pdf_visual_audit_report.md"


@dataclass
class PageMetrics:
    page: int
    width: int
    height: int
    ink_coverage: float
    bbox: tuple[int, int, int, int] | None


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command is not on PATH: {name}")
    return path


def page_count(pdf: Path) -> int:
    require_tool("pdfinfo")
    completed = subprocess.run(
        ["pdfinfo", str(pdf)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"^Pages:\s+(\d+)", completed.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read page count from pdfinfo output.")
    return int(match.group(1))


def render_pages(pdf: Path, out_dir: Path, pages: int, dpi: int) -> list[Path]:
    require_tool("pdftoppm")
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for page in range(1, pages + 1):
        stem = out_dir / f"page-{page:02d}"
        target = stem.with_suffix(".png")
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-q",
                "-r",
                str(dpi),
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                str(pdf),
                str(stem),
            ],
            check=True,
        )
        rendered.append(target)
    return rendered


def render_text(pdf: Path, out_dir: Path) -> list[str]:
    if not shutil.which("pdftotext"):
        return []
    text_path = out_dir / "main_layout.txt"
    subprocess.run(
        ["pdftotext", "-layout", str(pdf), str(text_path)],
        check=True,
        capture_output=True,
    )
    text = text_path.read_text(encoding="utf-8", errors="replace")
    return text.split("\f")


def page_metrics(path: Path, page: int) -> PageMetrics:
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        mask = gray.point(lambda pixel: 255 if pixel < 248 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            coverage = 0.0
        else:
            hist = mask.histogram()
            ink_pixels = hist[255]
            coverage = ink_pixels / float(width * height)
        return PageMetrics(page, width, height, coverage, bbox)


def suspicious(metrics: PageMetrics) -> list[str]:
    notes: list[str] = []
    if metrics.ink_coverage < 0.015:
        notes.append("very low ink coverage")
    if metrics.bbox is not None:
        left, top, right, bottom = metrics.bbox
        x_margin = min(left, metrics.width - right) / metrics.width
        y_margin = min(top, metrics.height - bottom) / metrics.height
        if x_margin < 0.025:
            notes.append("content is close to horizontal page edge")
        if y_margin < 0.025:
            notes.append("content is close to vertical page edge")
    return notes


def label_pages(text_pages: list[str]) -> tuple[set[int], dict[int, list[str]]]:
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }

    def clean_line(line: str) -> str:
        for old, new in ligatures.items():
            line = line.replace(old, new)
        return " ".join(line.split())

    selected: set[int] = set()
    labels: dict[int, list[str]] = {}
    for idx, text in enumerate(text_pages, start=1):
        page_labels = []
        for line in text.splitlines():
            clean = clean_line(line)
            if re.match(r"^(Table|Figure)\s+(?:[A-Z]\.)?\d+", clean):
                page_labels.append(clean[:140])
        if page_labels:
            selected.add(idx)
            labels[idx] = page_labels[:4]
    return selected, labels


def make_contact_sheet(
    images: list[Path],
    pages: list[int],
    out_path: Path,
    thumb_width: int = 190,
    columns: int = 5,
) -> None:
    if not pages:
        return
    thumbs = []
    for page in pages:
        with Image.open(images[page - 1]) as image:
            ratio = thumb_width / image.width
            thumb = image.resize((thumb_width, int(image.height * ratio)))
            canvas = Image.new("RGB", (thumb.width, thumb.height + 26), "white")
            canvas.paste(thumb.convert("RGB"), (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, thumb.height + 6), f"page {page}", fill=(30, 30, 30))
            thumbs.append(canvas)

    rows = (len(thumbs) + columns - 1) // columns
    cell_w = max(thumb.width for thumb in thumbs) + 14
    cell_h = max(thumb.height for thumb in thumbs) + 14
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * cell_w + 7
        y = (idx // columns) * cell_h + 7
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_report(
    report: Path,
    pdf: Path,
    out_dir: Path,
    metrics: list[PageMetrics],
    labels: dict[int, list[str]],
    suspicious_pages: dict[int, list[str]],
) -> None:
    table_pages = sorted(labels)
    status = "PASS" if not suspicious_pages else "WARN"
    all_sheet = out_dir / "contact_sheet_all.png"
    table_sheet = out_dir / "contact_sheet_tables_figures.png"
    lines = [
        "# PDF Visual Audit",
        "",
        f"PDF: `{pdf.relative_to(ROOT).as_posix()}`",
        f"Status: **{status}**",
        f"Pages rendered: {len(metrics)}",
        f"All-page contact sheet: `{all_sheet.relative_to(ROOT).as_posix()}`",
        f"Table/figure contact sheet: `{table_sheet.relative_to(ROOT).as_posix()}`",
        "",
        "This audit renders the compiled PDF and checks for blank pages or content close to page edges. "
        "It complements LaTeX log checks; it is not a substitute for a final human read.",
        "",
        "## Suspicious Pages",
        "",
    ]
    if suspicious_pages:
        for page, notes in suspicious_pages.items():
            lines.append(f"- Page {page}: {', '.join(notes)}.")
    else:
        lines.append("- None.")
    lines.extend(["", "## Table/Figure Pages", ""])
    if table_pages:
        for page in table_pages:
            joined = " | ".join(labels[page])
            lines.append(f"- Page {page}: {joined}")
    else:
        lines.append("- No table or figure captions detected by `pdftotext -layout`.")
    lines.extend(["", "## Page Metrics", ""])
    lines.append("| Page | Size | Ink coverage | Content box |")
    lines.append("| --- | ---: | ---: | --- |")
    for item in metrics:
        bbox = "none" if item.bbox is None else f"{item.bbox[0]},{item.bbox[1]}-{item.bbox[2]},{item.bbox[3]}"
        lines.append(f"| {item.page} | {item.width}x{item.height} | {item.ink_coverage:.3f} | {bbox} |")
    lines.append("")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render and audit the compiled manuscript PDF.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    pages = page_count(args.pdf)
    images = render_pages(args.pdf, args.out_dir, pages, args.dpi)
    metrics = [page_metrics(path, idx) for idx, path in enumerate(images, start=1)]
    suspicious_pages = {
        item.page: notes for item in metrics if (notes := suspicious(item))
    }
    text_pages = render_text(args.pdf, args.out_dir)
    table_pages, labels = label_pages(text_pages)
    selected_pages = sorted(set(range(1, pages + 1)))
    make_contact_sheet(images, selected_pages, args.out_dir / "contact_sheet_all.png", thumb_width=150, columns=6)
    make_contact_sheet(images, sorted(table_pages), args.out_dir / "contact_sheet_tables_figures.png", thumb_width=190, columns=4)
    write_report(args.report, args.pdf, args.out_dir, metrics, labels, suspicious_pages)
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
