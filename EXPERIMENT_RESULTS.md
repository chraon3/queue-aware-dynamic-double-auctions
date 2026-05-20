# Reproducibility Log

This file records the stable submission-facing replication status. Historical
pilot notes, smoke-test notes, and private review-tracking notes are not included
in this clean release package.

## Current manuscript

- Title: Queue Pressure and Trade Reduction in Dynamic Double Auctions
- Authors: Xingfu Zheng and Bo Wang
- Target journal: Journal of Economic Theory
- Manuscript source: `paper/main.tex`
- Generated tables: `paper/tables/`
- Generated figures: `paper/figures/`

## Stable check commands

```powershell
python -B run_submission_checks.py
python -B make_paper_assets.py
python -B redraw_main_figures.py
```

With Tectonic installed:

```powershell
python -B run_submission_checks.py --compile --visual
```

## Current local audit status

The clean package was built from the manuscript version that passed:

- manuscript quality audit
- table audit
- figure audit
- data schema audit
- PDF visual audit
- standalone LaTeX source compilation

## Notes on scope

The manuscript treats neural search as a mechanism-discovery tool. The reported
queue-aware rule is an audited candidate in a restricted McAfee-preserving class,
not a claim of full dynamic strategy-proofness or a global dynamic optimum.
