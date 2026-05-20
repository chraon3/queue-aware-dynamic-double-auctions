# Runbook

This runbook separates stable submission checks from heavier exploratory training
commands. It is intended for authors, referees, and research assistants who need
to rebuild the manuscript assets from the archived files.

## Environment

Create a Python 3.11 environment and install the listed dependencies:

```powershell
python -m pip install -r requirements.txt
```

Alternatively, use the conda environment file:

```powershell
conda env create -f environment.yml
conda activate queue-aware-double-auctions
```

## Fast Submission Check

Run the editorial and data audits:

```powershell
python -B run_submission_checks.py
```

The reports are written to `paper/submission_package/`:

- `quality_audit_report.md`
- `table_audit_report.md`
- `figure_audit_report.md`
- `data_schema_audit_report.md`

## Manuscript Build

The manuscript can be compiled with Tectonic. Install Tectonic on the system path
or place a portable executable at `.tools/tectonic/tectonic.exe`.

```powershell
python -B run_submission_checks.py --compile --visual
```

The compiled PDF is written to:

```text
paper/build/main.pdf
```

## Regenerate Tables and Figures

Regenerate tables and baseline figure files from stored experiment outputs:

```powershell
python -B make_paper_assets.py
```

Redraw the five main manuscript figures in journal style:

```powershell
python -B redraw_main_figures.py
```

## Full Dynamic Replication

The full dynamic suite is more computationally expensive than the fast submission
check. The main entry points are:

```powershell
python -B run_experiment_suite.py
python -B run_dynamic_suite.py
python -B run_dynamic_robustness.py
python -B evaluate_queue_aware_mcafee.py
python -B stress_queue_aware_mcafee.py
```

Specialized audit scripts are listed in `paper/submission_package/replication_note.md`.
