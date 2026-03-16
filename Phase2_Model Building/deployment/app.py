from __future__ import annotations

import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from inference import (
    get_available_datasets,
    get_available_models,
    get_available_regions,
    get_available_times,
    get_single_prediction,
    list_candidate_output_roots,
    load_deployment_data,
    topk_for_time,
)

st.set_page_config(page_title="IT5006 Crime Risk POC", layout="wide")

DEFAULT_FIXED_SEED = 0
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pretty_dataset_name(dataset: str) -> str:
    mapping = {
        "chicago_2025": "Chicago 2025 (In-domain)",
        "nibrs": "NIBRS (Out-of-domain)",
    }
    return mapping.get(str(dataset), str(dataset))


def _pretty_model_name(model: str) -> str:
    mapping = {
        "baseline_persistence": "Baseline Persistence",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "lstm": "LSTM",
    }
    return mapping.get(str(model), str(model))


def _friendly_region_label(region_id: str, dataset: str) -> str:
    rid = str(region_id)
    if dataset == "chicago_2025" and "CPD_DIST_" in rid:
        dist = rid.split("CPD_DIST_")[-1]
        dist = dist.lstrip("0") or "0"
        return f"Chicago District {dist} [{rid}]"
    if "__" in rid:
        state, code = rid.split("__", 1)
        return f"{state} Agency {code} [{rid}]"
    return rid


def _friendly_time_label(time_id: str, window_days: int) -> str:
    ts = pd.to_datetime(str(time_id), errors="coerce")
    if pd.isna(ts):
        return str(time_id)
    end = ts + timedelta(days=int(window_days) - 1)
    return f"{ts.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} [{ts.strftime('%Y-%m-%d')}]"


def _compact_area_label(friendly_label: str, max_len: int = 18) -> str:
    base = str(friendly_label).split("[")[0].strip()
    if len(base) <= int(max_len):
        return base
    return base[: max_len - 1] + "..."


@st.cache_data(show_spinner=False)
def _discover_output_roots(project_root: str):
    roots = list_candidate_output_roots(Path(project_root))
    return [str(p) for p in roots]


@st.cache_data(show_spinner=True)
def _load_data(project_root: str, output_root: str, fixed_seed: int):
    return load_deployment_data(
        project_root=Path(project_root),
        output_root=Path(output_root),
        fixed_seed=int(fixed_seed),
    )


st.title("IT5006 Predictive Policing - Deployment POC")
st.caption("Overall high-risk prediction for next window (seed fixed for deployment reproducibility)")

st.sidebar.header("Deployment Settings")
project_root = st.sidebar.text_input("Project root", value=str(PROJECT_ROOT))

candidate_roots = _discover_output_roots(project_root)
if not candidate_roots:
    st.error(
        "No valid output roots found. Expected folders like output_timeline_*/ with 7d/3d metrics/predictions.csv."
    )
    st.stop()

env_pref = os.getenv("PHASE2_OUTPUT_ROOT", "").strip()
default_idx = 0
if env_pref and env_pref in candidate_roots:
    default_idx = candidate_roots.index(env_pref)

output_root = st.sidebar.selectbox("Output root", options=candidate_roots, index=default_idx)

fixed_seed = DEFAULT_FIXED_SEED
st.sidebar.text_input("Fixed seed (locked)", value=str(fixed_seed), disabled=True)

try:
    data = _load_data(project_root, output_root, fixed_seed)
except Exception as exc:
    st.error(f"Failed to load deployment data: {exc}")
    st.stop()

st.sidebar.success(f"Loaded: {data.output_root.name}")

window_label = st.sidebar.radio("Time window", options=["7d", "3d"], horizontal=True)
window_days = 7 if window_label == "7d" else 3

datasets = get_available_datasets(data, window_days)
if not datasets:
    st.error("No datasets available for the selected time window.")
    st.stop()

dataset = st.sidebar.selectbox("Dataset", options=datasets)

models = get_available_models(data, window_days, dataset)
if not models:
    st.error("No models available for selected window + dataset.")
    st.stop()

model = st.sidebar.selectbox("Model", options=models)

regions = get_available_regions(data, window_days, dataset, model)
times = get_available_times(data, window_days, dataset, model)

if not regions or not times:
    st.error("No region/time options found for selected configuration.")
    st.stop()

st.info(
    "Selection guide: pick **Area/Jurisdiction** and **Current Window Start Date**. "
    "The app then shows the model's prediction for the **next window (t+1)**."
)

col_a, col_b = st.columns(2)
with col_a:
    region_id = st.selectbox("Area / Jurisdiction", options=regions, format_func=lambda rid: _friendly_region_label(rid, dataset))
with col_b:
    time_id = st.selectbox("Current Window Start Date", options=times, format_func=lambda tid: _friendly_time_label(tid, window_days))

predict_clicked = st.button("Run Prediction", type="primary")

if predict_clicked:
    st.session_state["last_query"] = {
        "window_days": int(window_days),
        "dataset": str(dataset),
        "model": str(model),
        "region_id": str(region_id),
        "time_id": str(time_id),
    }

st.markdown("---")

threshold = data.thresholds.get(window_days)
if threshold is None:
    st.info("Threshold not found in run summary. Using stored y_pred from model output.")
else:
    st.info(f"Label rule for this window: HighRisk = 1 if next-window incidents >= {threshold} (estimated on Chicago train only).")

active_query = st.session_state.get("last_query")

if active_query:
    q_window = int(active_query["window_days"])
    q_dataset = str(active_query["dataset"])
    q_model = str(active_query["model"])
    q_region = str(active_query["region_id"])
    q_time = str(active_query["time_id"])

    st.caption(
        "Current displayed query: "
        f"window={q_window}d, dataset={q_dataset}, model={q_model}, area={q_region}, time={q_time}"
    )

    if st.button("Clear Displayed Result"):
        st.session_state.pop("last_query", None)
        st.rerun()

    try:
        row = get_single_prediction(
            data=data,
            window_days=q_window,
            dataset=q_dataset,
            model=q_model,
            region_id=q_region,
            time_id=q_time,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Risk score",
        f"{float(row['y_score']):.4f}",
        help="Model risk output for the selected area and current window. Higher means higher predicted risk for next window.",
    )
    c2.metric(
        "Predicted HighRisk",
        int(row["y_pred"]),
        help="Binary decision from risk score using deployment threshold. 1 = predicted high-risk, 0 = not high-risk.",
    )
    c3.metric(
        "Observed next-window incidents",
        int(round(float(row["y_count"]))),
        help="Historical realized incidents in the next window. Shown for retrospective validation/demo only.",
    )
    c4.metric(
        "Observed HighRisk (backtest)",
        int(row["y"]),
        help="Historical ground-truth high-risk label for the next window, derived from threshold rule.",
    )

    st.caption(
        "Observed values are shown for retrospective validation/demo only. "
        "In live operations, future outcomes are unknown at prediction time."
    )

    st.subheader("Selected Record")
    record_df = pd.DataFrame(
        [
            {
                "Time Window": f"{q_window}d",
                "Dataset": _pretty_dataset_name(q_dataset),
                "Model": _pretty_model_name(q_model),
                "Area / Jurisdiction": _friendly_region_label(q_region, q_dataset),
                "Current Window Start": str(q_time),
                "Risk Score": float(row["y_score"]),
                "Predicted HighRisk": int(row["y_pred"]),
                "Observed Next-window Incidents": float(row["y_count"]),
                "Observed HighRisk (backtest)": int(row["y"]),
                "Seed (fixed)": int(row["seed"]),
            }
        ]
    )
    st.dataframe(record_df, use_container_width=True)

    st.markdown("#### Selected Area Trend Over Time")
    hist = data.frame[
        (data.frame["time_window_days"] == q_window)
        & (data.frame["dataset"] == q_dataset)
        & (data.frame["model"] == q_model)
        & (data.frame["region_id"] == q_region)
    ].copy()
    hist = hist.sort_values("time_id").reset_index(drop=True)

    if len(hist) > 0:
        hist["time_id"] = pd.to_datetime(hist["time_id"], errors="coerce")
        hist = hist.dropna(subset=["time_id"]).copy()
        hist["time_label"] = hist["time_id"].dt.strftime("%Y-%m-%d")

        left, right = st.columns(2)
        with left:
            st.caption("Risk score trajectory")
            st.line_chart(hist.set_index("time_label")[["y_score"]], use_container_width=True)
        with right:
            st.caption("Observed next-window incidents (backtest)")
            st.line_chart(hist.set_index("time_label")[["y_count"]], use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.plot(hist["time_label"], hist["y_score"], linewidth=1.4, label="Risk score")
        ax.scatter(
            hist["time_label"],
            hist["y_score"],
            c=hist["y_pred"],
            s=10,
            cmap="coolwarm",
            alpha=0.8,
            label="Predicted HighRisk (color)",
        )
        ax.set_title("Risk Score Timeline (selected area)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelrotation=45)
        step = max(1, len(hist) // 8)
        for i, lbl in enumerate(ax.get_xticklabels()):
            if i % step != 0:
                lbl.set_visible(False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Top-K Operational Ranking (All Areas at the Same Time)")
    st.caption(
        "This section ranks **all areas/jurisdictions** at the selected time "
        "(same window + dataset + model). It simulates limited-resource deployment."
    )

    all_regions = get_available_regions(data, q_window, q_dataset, q_model)
    n_regions = len(all_regions)
    default_k = 10 if n_regions >= 10 else n_regions
    slider_key = f"topk_{q_window}_{q_dataset}_{q_model}_{q_time}"
    k = st.slider("K", min_value=1, max_value=n_regions, value=default_k, key=slider_key)

    try:
        topk_df, stats = topk_for_time(
            data=data,
            window_days=q_window,
            dataset=q_dataset,
            model=q_model,
            time_id=q_time,
            k=k,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Coverage@K",
        f"{stats['coverage_at_k']:.3f}",
        help="Share of total next-window incidents covered by the top-K ranked areas.",
    )
    s2.metric(
        "Top-K hit rate",
        f"{stats['topk_hit_rate']:.3f}",
        help="Fraction of top-K areas that are truly high-risk in backtest labels.",
    )
    s3.metric(
        "Top-K incidents / total",
        f"{int(stats['topk_incidents'])} / {int(stats['total_incidents'])}",
        help="Absolute incident count captured by top-K areas versus total incidents at that time.",
    )

    topk_show = topk_df.copy()
    topk_show.insert(
        1,
        "Area / Jurisdiction",
        topk_show["region_id"].astype(str).map(lambda rid: _friendly_region_label(rid, q_dataset)),
    )
    topk_show = topk_show.rename(
        columns={
            "rank": "Rank",
            "region_id": "Region ID (raw)",
            "y_score": "Risk Score",
            "y_pred": "Predicted HighRisk",
            "y_count": "Observed Next-window Incidents",
            "y": "Observed HighRisk (backtest)",
        }
    )
    st.dataframe(topk_show, use_container_width=True)

    st.markdown("#### Top-K Risk Score Curve")
    topk_curve = topk_show[["Rank", "Area / Jurisdiction", "Region ID (raw)", "Risk Score"]].copy().sort_values("Rank")
    topk_curve["axis_label"] = topk_curve.apply(
        lambda r: f"{int(r['Rank'])}-{_compact_area_label(r['Area / Jurisdiction'])}", axis=1
    )

    fig_topk, ax_topk = plt.subplots(figsize=(10, 3.2))
    x = list(range(len(topk_curve)))
    y = topk_curve["Risk Score"].to_list()
    ax_topk.plot(x, y, marker="o", linewidth=1.8)

    highlight = topk_curve[topk_curve["Region ID (raw)"] == q_region]
    if not highlight.empty:
        h_idx = int(highlight.index[0])
        ax_topk.scatter([h_idx], [y[h_idx]], color="red", s=60, zorder=5)
        ax_topk.annotate("selected area", (h_idx, y[h_idx]), textcoords="offset points", xytext=(0, 8), ha="center", color="red")

    ax_topk.set_title("Top-K Risk Score by Rank and Area")
    ax_topk.set_xlabel("Priority Rank - Area")
    ax_topk.set_ylabel("Risk Score")
    ax_topk.grid(alpha=0.25)

    step = 1 if len(topk_curve) <= 18 else max(2, len(topk_curve) // 10)
    tick_pos = [i for i in x if i % step == 0]
    tick_lbl = [topk_curve.iloc[i]["axis_label"] for i in tick_pos]
    ax_topk.set_xticks(tick_pos)
    ax_topk.set_xticklabels(tick_lbl, rotation=40, ha="right")

    fig_topk.tight_layout()
    st.pyplot(fig_topk, use_container_width=True)
    plt.close(fig_topk)

    selected_rank = topk_df.loc[topk_df["region_id"] == q_region, "rank"]
    if len(selected_rank) > 0:
        st.success(f"Selected region rank at this time: #{int(selected_rank.iloc[0])}")
    else:
        st.warning("Selected region is outside Top-K for the chosen time/model.")
else:
    st.write("Choose inputs and click **Run Prediction** to lock a query and view results.")

st.markdown("---")
st.caption(
    f"Deployment mode: fixed seed = {DEFAULT_FIXED_SEED}. "
    "Error handling includes missing data checks, invalid option guards, and no-match validation."
)
