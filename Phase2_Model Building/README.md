# IT5006 Predictive Policing (Phase 2)

A reproducible end-to-end analytics pipeline for **spatiotemporal crime risk prediction**:

- data ingestion and schema harmonization
- feature engineering
- model training (4 models)
- in-domain and out-of-domain evaluation
- paper-style tables and figures
- auto-generated reports

This repository is built for the IT5006 Phase/Milestone 2 deliverable.

## 1) Problem Definition (Current Pipeline)

The current implemented task is:

- **Unit of prediction**: `(region_id, time_id)`
- **Target**: overall high-risk in the **next window** (`t+1`), not crime-type-specific labels
- **Label**:
  - `y_count = total incident count at t+1`
  - `y = 1(y_count >= threshold)`
- **Threshold rule**:
  - estimated from **Chicago train only** (2015-2024)
  - computed separately for each window setting (7d vs 3d)
  - fixed for Chicago 2025 and NIBRS evaluation (no refit)

`crime_group` is still used for harmonization and composition features, but the main label is overall risk.

## 2) Data Inputs

Place raw files under `data/raw/`.

Required Chicago files (auto-detected by filename pattern):
- `crime_train_2015_2024.csv`
- `crime_test_val_2025.csv`

Required NIBRS tables (auto-detected recursively):
- `NIBRS_incident.csv`
- `NIBRS_OFFENSE.csv`
- `NIBRS_OFFENSE_TYPE.csv`
- `agencies.csv`

The pipeline automatically scans and links valid NIBRS roots.

## 3) Environment Setup

Recommended: Python 3.8+.

```bash
cd "/Users/tim/Desktop/NUS/semester 2/IT5006/Group project/phase2"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Run the Pipeline

### Full run (both 7d and 3d, all 4 models)

```bash
python -m src.run_all --project_root "/Users/tim/Desktop/NUS/semester 2/IT5006/Group project/phase2"
```

### Run a single window

```bash
python -m src.run_all --project_root "<project_root>" --time_window_days 7
python -m src.run_all --project_root "<project_root>" --time_window_days 3
```

### Select models manually

Supported aliases: `all | baseline | logistic | rf | lstm`

```bash
python -m src.run_all --project_root "<project_root>" --models baseline logistic rf lstm
python -m src.run_all --project_root "<project_root>" --models rf lstm
```

### Custom output folder name

```bash
python -m src.run_all \
  --project_root "<project_root>" \
  --output_root_name output_timeline_YYYYMMDD_HHMMSS
```

## 5) Outputs

Each run writes artifacts under the selected output root (default: `output_timeline_<timestamp>`):

- `7d/`
- `3d/`
- `compare_windows/`

Per-window structure:
- `data_processed/`: `Xy_chicago_train`, `Xy_chicago_2025`, `Xy_nibrs`, mapping files
- `models/seed_{0,1,2}/`: trained model artifacts and logs
- `metrics/`: predictions, `metrics_long.csv`, top-k/temporal/spatial metrics
- `tables/`: Table 1 and Table 2
- `figures/`: Figure 1/3/4/6 and geo-supporting plots
- `logs/`: run summaries and schema checks

Cross-window outputs:
- `compare_windows/figures/fig2_gap_auc.png`
- `compare_windows/figures/fig2_gap_f1.png`
- `compare_windows/figures/fig_compare_windows_overall_{auc,f1,precision,recall}.png`

Reports:
- `<output_root>/report_phase2.md`
- `<output_root>/{7d,3d}/report_phase2.md`

## 6) Core Model Set

1. `baseline_persistence`
2. `logistic_regression` (PyTorch linear classifier)
3. `random_forest` (custom torch-based implementation)
4. `lstm` (PyTorch)

All four are runnable from the same pipeline entrypoint.

## 7) Project Structure

```text
src/
  config.py
  data_io.py
  preprocess_chicago.py
  preprocess_nibrs.py
  mapping.py
  features.py
  train.py
  eval.py
  paper_outputs.py
  report.py
  run_all.py
```

## 8) Reproducibility Notes

- Fixed evaluation seeds: `0, 1, 2`
- 3-day anchor date: `2015-01-01` (configurable)
- Threshold fit domain: **Chicago train only**
- Scaler/statistics fit domain: **Chicago train only**

## 9) Common Issue

If you see:
- `ImportError: attempted relative import with no known parent package`

Run from project root with module mode:

```bash
python -m src.run_all --project_root "<project_root>"
```

(not `python -m run_all`).
