# IT5006 Group 6 Predictive Policing Pipeline

This repository contains the Phase 2 and Phase 3 implementation for an IT5006 course project on cross-domain crime risk prediction.

The pipeline trains on Chicago historical crime data, evaluates on:
- Chicago 2025 (in-domain)
- NIBRS (out-of-domain)

It supports:
- two time windows (`7d` and `3d`)
- four models (`baseline_persistence`, `logistic_regression`, `random_forest`, `lstm`)
- automated metrics/tables/figures/report generation
- a Streamlit deployment POC for single-area prediction and Top-K operational ranking

## Repository Layout

```text
project-root/
  src/                    # Training, feature engineering, evaluation, reporting
  deployment/             # Streamlit app and deployment helpers
  docs/                   # Project phase reports (PDF)
  notesbook/              # EDA notebook
```

## Environment Setup

Python `3.10+` is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib torch pyarrow streamlit
```

## Expected Data Layout

The pipeline auto-discovers input files under `data/raw/`.

Required Chicago files (name-based detection):
- one CSV whose filename contains both `crime_train` and `2015`
- one CSV whose filename contains both `crime_test` and `2025`

Required NIBRS files (folder-based detection, can have multiple roots):
- `NIBRS_incident.csv`
- `NIBRS_OFFENSE.csv`
- `NIBRS_OFFENSE_TYPE.csv`
- `agencies.csv`

Example:

```text
data/
  raw/
    chicago/
      chicago_crime_train_2015_2024.csv
      chicago_crime_test_2025.csv
    nibrs_state_a/
      NIBRS_incident.csv
      NIBRS_OFFENSE.csv
      NIBRS_OFFENSE_TYPE.csv
      agencies.csv
    nibrs_state_b/
      NIBRS_incident.csv
      NIBRS_OFFENSE.csv
      NIBRS_OFFENSE_TYPE.csv
      agencies.csv
```

## Run the Full Pipeline

Run from repository root:

```bash
python -m src.run_all --project_root .
```

Useful options:

```bash
# Choose models
python -m src.run_all --project_root . --models logistic rf

# Run only one time window
python -m src.run_all --project_root . --time_window_days 7

# Set custom output folder name
python -m src.run_all --project_root . --output_root_name my_outputs

# Tune LSTM sequence length and 3-day anchor date
python -m src.run_all --project_root . --lstm_sequence_length 8 --anchor_date 2015-01-01
```

## Output Artifacts

By default, each run creates:

```text
output_timeline_<YYYYMMDD_HHMMSS>/
  7d/
    data_processed/
    models/
    metrics/
    tables/
    figures/
    logs/
    report_phase2.md
  3d/
    data_processed/
    models/
    metrics/
    tables/
    figures/
    logs/
    report_phase2.md
  compare_windows/
    figures/
  report_phase2.md         # cross-window comparison report
```

Key files you will typically inspect:
- `*/metrics/predictions.csv`
- `*/metrics/metrics_overall.csv`
- `*/metrics/metrics_delta_auc.csv`
- `*/logs/run_summary.json`
- `report_phase2.md` (window-level and cross-window reports)

## Deployment Dashboard (Phase 3 POC)

The Streamlit app reads generated pipeline outputs and serves an interactive prediction dashboard.

Run locally:

```bash
streamlit run deployment/app.py
```

Optional: force a specific output root:

```bash
export PHASE2_OUTPUT_ROOT="/absolute/path/to/output_timeline_20260308_194946"
streamlit run deployment/app.py
```

Main capabilities:
- single query prediction by area and time
- risk score and predicted high-risk label
- retrospective observed outcomes (for backtest/demo)
- Top-K operational ranking with Coverage@K and hit rate

## Notes

- This repository currently focuses on code and documentation. Raw datasets are expected to be prepared under `data/raw/`.
- Use `python -m src.run_all` (module mode) from project root to avoid relative-import issues.
- The deployment app uses fixed `seed=0` for reproducible UI behavior.
