# Dashboard Usage Guide

## 1. Purpose
This dashboard is a POC/MVP interface for **overall next-window crime risk prediction**.
It is designed for demonstration and operational prioritization, not real-time retraining.

## 2. What You Need Before Using
1. Pipeline outputs already generated (contains predictions):
- `output_timeline_*/7d/metrics/predictions.csv`
- `output_timeline_*/3d/metrics/predictions.csv`

2. Dashboard started successfully:
```bash
streamlit run deployment/app.py
```

## 3. Main Workflow
### Step 1: Deployment Settings (left sidebar)
1. `Project root`: root path of this repository.
2. `Output root`: choose one run folder (for example `output_timeline_20260308_194946`).
3. `Fixed seed (locked)`: fixed to `0` for reproducibility.
4. `Time window`: choose `7d` or `3d`.
5. `Dataset`: choose `Chicago 2025` (in-domain) or `NIBRS` (out-of-domain).
6. `Model`: choose one of the four trained models.

### Step 2: Query Input (main area)
1. `Area / Jurisdiction`: select district/agency.
2. `Current Window Start Date`: select current prediction window.
3. Click `Run Prediction`.

### Step 3: Read Prediction Outputs
After clicking `Run Prediction`, the dashboard shows:
1. `Risk score`: model risk output for this area and current window.
2. `Predicted HighRisk`: binary decision (`1` high-risk, `0` not high-risk).
3. `Observed next-window incidents`: backtest ground truth count at `t+1`.
4. `Observed HighRisk (backtest)`: backtest binary label at `t+1`.

Note: observed values are shown for retrospective validation/demo only.

## 4. Visualization Blocks
### 4.1 Selected Record
A one-row summary table for the selected area-time query.

### 4.2 Selected Area Trend Over Time
Shows this area's trend under the selected window/dataset/model:
1. Risk score trajectory.
2. Observed next-window incidents trajectory (backtest).
3. Risk timeline plot with highlighted predicted high-risk points.

### 4.3 Top-K Operational Ranking (same time)
This block changes perspective from one area to **all areas at the selected time**.

1. `K` slider controls operational capacity.
2. Top-K table lists highest-priority areas by risk score.
3. Metrics:
- `Coverage@K`: fraction of incidents covered by top-K areas.
- `Top-K hit rate`: fraction of top-K areas that are truly high-risk.
- `Top-K incidents / total`: absolute coverage count.
4. `Top-K Risk Score Curve`: rank-based risk curve with area labels and selected-area highlight.

## 5. Error Handling
Built-in checks include:
1. Missing output folders/files.
2. Missing required columns.
3. Invalid combination of window/dataset/model/area/time.
4. No rows available after fixed-seed filtering.
5. Top-K calculation failure when selected slice is empty.

If an error appears, first verify selected `Output root` and dataset/model options.

## 6. Suggested Demo Script (2-3 minutes)
1. Pick `7d + chicago_2025 + random_forest`.
2. Select one district and date, run prediction.
3. Explain the 4 key metrics (risk score, y_pred, observed y_count, observed y).
4. Show area trend block.
5. Move to Top-K, change K from 5 to 10, explain Coverage@K shift.
6. Point to selected area rank in Top-K.

## 7. For Final Report (required evidence)
Include:
1. Live application URL.
2. Platform and justification.
3. Usage steps.
4. Screenshots:
- Home + inputs
- Single prediction result
- Trend block
- Top-K block
- One error-handling example

## 8. Cloud Deployment Path FAQ
### Q: Do I need to change file paths for cloud deployment?
Short answer: **usually no hardcoded local path should be kept**.

Current app behavior:
1. It auto-detects project root from app location.
2. It scans output folders like `output_timeline_*` under repo root.
3. You can override output root with env var `PHASE2_OUTPUT_ROOT`.

What to do in cloud:
1. Do **not** use absolute local paths like `/Users/tim/...`.
2. Ensure prediction outputs are available in the deployed repo (or accessible storage).
3. If outputs are not in repo, app cannot load data and must be adapted to read from external storage/API.
