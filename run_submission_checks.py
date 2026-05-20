from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKS = [
    ["quality_audit.py"],
    ["table_audit.py"],
    ["figure_audit.py"],
    ["data_schema_audit.py"],
]


def run(command: list[str], cwd: Path) -> int:
    printable = " ".join(command)
    print(f"\n==> {printable}")
    completed = subprocess.run(command, cwd=cwd, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run submission-facing quality checks.")
    parser.add_argument("--compile", action="store_true", help="Also compile paper/main.tex with Tectonic.")
    parser.add_argument("--visual", action="store_true", help="Render the compiled PDF and create visual audit contact sheets.")
    parser.add_argument("--strict", action="store_true", help="Pass --strict to audit scripts.")
    args = parser.parse_args()

    failures = 0
    for check in CHECKS:
        command = [sys.executable, "-B", *check]
        if args.strict:
            command.append("--strict")
        failures += int(run(command, ROOT) != 0)

    if args.compile:
        bundled = ROOT / ".tools" / "tectonic" / "tectonic.exe"
        tectonic = str(bundled) if bundled.exists() else shutil.which("tectonic")
        if not tectonic:
            print("Missing Tectonic. Install it on PATH or place it at .tools/tectonic/tectonic.exe.")
            failures += 1
        else:
            failures += int(
                run(
                    [tectonic, "main.tex", "--outdir", "build", "--keep-logs", "--keep-intermediates"],
                    ROOT / "paper",
                )
                != 0
            )

    if args.visual:
        failures += int(run([sys.executable, "-B", "pdf_visual_audit.py"], ROOT) != 0)

    if failures:
        print(f"\nCompleted with {failures} failing check(s).")
        return 1
    print("\nAll requested checks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
