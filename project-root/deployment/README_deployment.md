# Deployment Guide (Phase 3 POC/MVP)

This folder contains a lightweight deployment app for **overall high-risk prediction**.

## What this app does

- Loads already-generated prediction outputs from the pipeline
- Fixes `seed=0` for reproducible deployment behavior
- Lets users choose:
  - time window (`7d` or `3d`)
  - dataset (`chicago_2025` or `nibrs`)
  - model (`baseline_persistence`, `logistic_regression`, `random_forest`, `lstm`)
  - `region_id` and `time_id`
- Returns:
  - risk score (`y_score`)
  - predicted high-risk label (`y_pred`)
  - historical observed outcomes (`y_count`, `y`) for backtest demo
  - Top-K table and deployment-style metrics (Coverage@K, Top-K hit rate)

## Files

- `deployment/app.py`: Streamlit UI
- `deployment/inference.py`: data loading and prediction helper logic
- `deployment/Usage guide.md`: user-facing dashboard usage guide
- `deployment/assets/screenshots/`: report screenshots

## Prerequisites

1. You must have pipeline outputs already generated, e.g.:

- `output_timeline_*/7d/metrics/predictions.csv`
- `output_timeline_*/3d/metrics/predictions.csv`

2. Install dependencies from project root:

```bash
pip install -r requirements.txt
```

## Run locally

From project root:

```bash
streamlit run deployment/app.py
```

Optional: pin a specific output root via environment variable:

```bash
export PHASE2_OUTPUT_ROOT="/abs/path/to/output_timeline_20260308_194946"
streamlit run deployment/app.py
```

## Streamlit Cloud deployment

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app.
3. Set app entrypoint to: `deployment/app.py`.
4. Ensure `requirements.txt` is present in repository root.
5. Deploy and copy the live URL.

## Error handling included

- Missing output folders/files
- Missing required columns
- Empty data after fixed-seed filtering
- Invalid combination of window/dataset/model/region/time
- Missing rows for Top-K computation
