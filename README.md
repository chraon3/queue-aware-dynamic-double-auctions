# Queue-Aware Dynamic Double Auctions

[![DOI](https://zenodo.org/badge/1244350843.svg)](https://doi.org/10.5281/zenodo.20313914)

This repository contains the replication package for the manuscript:

**Queue Pressure and Trade Reduction in Dynamic Double Auctions**

Authors: Xingfu Zheng and Bo Wang, Harbin Engineering University.

Recommended repository name: `queue-aware-dynamic-double-auctions`.

Versioned DOI for the JET submission release:
https://doi.org/10.5281/zenodo.20314091

GitHub release:
https://github.com/chraon3/queue-aware-dynamic-double-auctions/releases/tag/v1.0.2-jet-submission

## Contents

- `paper/main.tex`: manuscript source.
- `paper/tables/`: generated CSV and LaTeX tables used by the manuscript.
- `paper/figures/`: generated figure assets, including vector PDF versions used in the manuscript.
- `paper/submission_package/replication_note.md`: concise replication note.
- `RUNBOOK.md`: fast checks and asset-generation workflow.
- `*.py`: experiment, audit, and paper-asset scripts.
- `experiments/`: stored experiment outputs used to regenerate tables and audits.
- `experiments_summary/`: compact summary outputs.

## Fast Check

```powershell
python -B run_submission_checks.py
```

To include a LaTeX build, install Tectonic or place a Tectonic executable at
`.tools/tectonic/tectonic.exe`, then run:

```powershell
python -B run_submission_checks.py --compile --visual
```

## Dependencies

Use Python 3.11. The main dependencies are listed in `requirements.txt` and
`environment.yml`.

## AI-Assisted Preparation

The associated manuscript discloses the use of OpenAI ChatGPT/Codex for language
editing, manuscript organization, code/documentation drafting, and reproducibility
checks. The authors reviewed and edited all tool-assisted material and take full
responsibility for the content.

## License

The replication code is released under the MIT License unless otherwise specified.
Generated manuscript text and third-party references remain governed by their
respective copyrights.

