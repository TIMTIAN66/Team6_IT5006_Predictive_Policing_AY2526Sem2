from __future__ import annotations

import argparse
from pathlib import Path
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .mapping import CRIME_GROUPS


def _extract_region_state(region_series: pd.Series) -> pd.Series:
    state = region_series.fillna("UNK").astype(str).str.strip().str.upper()
    state = state.str.split("__", n=1).str[0]
    state = state.replace({"": "UNK", "NAN": "UNK"})
    return state.fillna("UNK")


def _window_frequency(time_window_days: int) -> str:
    if int(time_window_days) == 7:
        return "W-MON"
    if int(time_window_days) == 3:
        return "3D"
    raise ValueError(f"Unsupported time_window_days={time_window_days}. Expected 3 or 7.")


def to_week_start(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce")
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def make_time_id(
    date_series: pd.Series,
    time_window_days: int = 7,
    anchor_date: str = "2015-01-01",
) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce").dt.normalize()
    window_days = int(time_window_days)

    if window_days == 7:
        return to_week_start(dt)

    if window_days == 3:
        anchor = pd.Timestamp(anchor_date).normalize()
        out = pd.Series(pd.NaT, index=dt.index, dtype="datetime64[ns]")
        valid = dt.notna()
        if valid.any():
            day_delta = (dt.loc[valid] - anchor).dt.days.astype(np.int64)
            idx = np.floor_divide(day_delta, 3).astype(np.int64)
            out.loc[valid] = anchor + pd.to_timedelta(idx * 3, unit="D")
        return out

    raise ValueError(f"Unsupported time_window_days={time_window_days}. Expected 3 or 7.")


def aggregate_weekly_events(
    events: pd.DataFrame,
    time_window_days: int = 7,
    anchor_date: str = "2015-01-01",
) -> pd.DataFrame:
    working = events.copy()
    working["time_id"] = make_time_id(
        working["event_date"],
        time_window_days=time_window_days,
        anchor_date=anchor_date,
    )

    agg_total = (
        working.groupby(["region_id", "time_id"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    if agg_total.empty:
        out_cols = ["region_id", "time_id", "count"] + [f"count_grp_{g}" for g in CRIME_GROUPS]
        return pd.DataFrame(columns=out_cols)

    agg_group = (
        working.groupby(["region_id", "time_id", "crime_group"], dropna=False)
        .size()
        .rename("group_count")
        .reset_index()
    )

    regions = sorted(agg_total["region_id"].dropna().unique().tolist())
    freq = _window_frequency(time_window_days)
    weeks = pd.date_range(agg_total["time_id"].min(), agg_total["time_id"].max(), freq=freq)

    grid = pd.MultiIndex.from_product([regions, weeks], names=["region_id", "time_id"]).to_frame(index=False)

    panel = grid.merge(agg_total, on=["region_id", "time_id"], how="left")
    panel["count"] = panel["count"].fillna(0).astype(int)

    group_panel = (
        agg_group.pivot_table(
            index=["region_id", "time_id"],
            columns="crime_group",
            values="group_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=CRIME_GROUPS, fill_value=0)
        .reset_index()
    )
    group_panel = group_panel.rename(columns={g: f"count_grp_{g}" for g in CRIME_GROUPS})
    panel = panel.merge(group_panel, on=["region_id", "time_id"], how="left")

    for g in CRIME_GROUPS:
        col = f"count_grp_{g}"
        if col not in panel.columns:
            panel[col] = 0
        panel[col] = panel[col].fillna(0).astype(float)

    return panel.sort_values(["region_id", "time_id"]).reset_index(drop=True)


def build_supervised_weekly(
    panel: pd.DataFrame,
    time_window_days: int = 7,
) -> pd.DataFrame:
    frame = panel.copy().sort_values(["region_id", "time_id"])
    frame["region_state"] = _extract_region_state(frame["region_id"])

    group_keys = ["region_id"]
    grouped = frame.groupby(group_keys, group_keys=False)

    frame["lag_0"] = frame["count"]
    frame["lag_1"] = grouped["count"].shift(1)
    frame["lag_2"] = grouped["count"].shift(2)
    frame["lag_4"] = grouped["count"].shift(4)

    frame["rolling_mean_4"] = grouped["count"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    frame["rolling_sum_4"] = grouped["count"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
    frame["ewma_4"] = grouped["count"].transform(lambda s: s.shift(1).ewm(span=4, adjust=False).mean())

    # Spatial proxy features: within-state peer activity (excluding the current region).
    # This keeps Chicago/NIBRS feature logic aligned without requiring external GIS files.
    # Use tuple-style named aggregation for pandas compatibility across versions.
    state_time = frame.groupby(["region_state", "time_id"], as_index=False).agg(
        state_total_count=("count", "sum"),
        state_region_n=("count", "count"),
    )
    frame = frame.merge(state_time, on=["region_state", "time_id"], how="left")
    denom = np.maximum(frame["state_region_n"] - 1, 1)
    frame["neighbor_lag_0"] = np.where(
        frame["state_region_n"] > 1,
        (frame["state_total_count"] - frame["count"]) / denom,
        0.0,
    )
    grouped_after_neighbor = frame.groupby(group_keys, group_keys=False)
    frame["neighbor_lag_1"] = grouped_after_neighbor["neighbor_lag_0"].shift(1)
    frame["neighbor_roll_mean_4"] = grouped_after_neighbor["neighbor_lag_0"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    frame["neighbor_ewma_4"] = grouped_after_neighbor["neighbor_lag_0"].transform(
        lambda s: s.shift(1).ewm(span=4, adjust=False).mean()
    )
    frame["self_neighbor_ratio_lag0"] = np.where(
        frame["neighbor_lag_0"] > 0,
        frame["lag_0"] / frame["neighbor_lag_0"],
        frame["lag_0"],
    )

    frame["y_count"] = grouped["count"].shift(-1)
    frame["next_time"] = frame["time_id"] + pd.Timedelta(days=int(time_window_days))

    frame["month"] = frame["time_id"].dt.month
    frame["weekofyear"] = frame["time_id"].dt.isocalendar().week.astype(int)
    frame["quarter"] = frame["time_id"].dt.quarter
    frame["sin_weekofyear"] = np.sin(2 * np.pi * frame["weekofyear"] / 52.0)
    frame["cos_weekofyear"] = np.cos(2 * np.pi * frame["weekofyear"] / 52.0)

    grp_cols = [c for c in frame.columns if c.startswith("count_grp_")]
    lag_cols: List[str] = []
    for col in grp_cols:
        lag_col = f"{col}_lag1"
        frame[lag_col] = grouped[col].shift(1)
        lag_cols.append(lag_col)

    if lag_cols:
        lag_sum = frame[lag_cols].sum(axis=1)
        for col in grp_cols:
            lag_col = f"{col}_lag1"
            suffix = col.replace("count_grp_", "")
            share_col = f"lag1_share_{suffix}"
            frame[share_col] = np.where(lag_sum > 0, frame[lag_col] / lag_sum, 0.0)

    required = ["lag_1", "lag_2", "lag_4", "neighbor_lag_1", "y_count"]
    frame = frame.dropna(subset=required).copy()

    return frame.reset_index(drop=True)


def split_chicago_train_test(
    supervised_all: pd.DataFrame,
    chicago_test_start: pd.Timestamp,
    chicago_test_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (supervised_all["time_id"] < chicago_test_start) & (
        supervised_all["next_time"] < chicago_test_start
    )
    test_mask = (supervised_all["time_id"] >= chicago_test_start) & (
        supervised_all["next_time"] <= chicago_test_end
    )

    train_df = supervised_all.loc[train_mask].copy()
    test_df = supervised_all.loc[test_mask].copy()
    return train_df, test_df


def compute_threshold(chicago_train_df: pd.DataFrame, quantile: float) -> int:
    q = chicago_train_df["y_count"].quantile(quantile)
    return int(max(1, np.ceil(q)))


def compute_threshold_by_group(
    chicago_train_df: pd.DataFrame,
    quantile: float,
    group_col: str = "crime_group",
) -> Dict[str, int]:
    if group_col not in chicago_train_df.columns:
        raise ValueError(f"Missing group column for thresholding: {group_col}")
    if "y_count" not in chicago_train_df.columns:
        raise ValueError("Missing y_count column for thresholding.")

    thresholds: Dict[str, int] = {}
    grouped = chicago_train_df.groupby(group_col)["y_count"]
    for group, series in grouped:
        q = float(series.quantile(quantile))
        thresholds[str(group)] = int(max(1, math.ceil(q)))
    return thresholds


def add_binary_label(
    df: pd.DataFrame,
    threshold: Union[int, Mapping[str, int]],
    group_col: str = "crime_group",
    default_threshold: int | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    if isinstance(threshold, Mapping):
        if group_col not in frame.columns:
            raise ValueError(f"Missing group column for threshold mapping: {group_col}")
        mapped = frame[group_col].astype(str).map({str(k): int(v) for k, v in threshold.items()})
        if default_threshold is None:
            default_threshold = int(max([1] + [int(v) for v in threshold.values()]))
        threshold_series = mapped.fillna(int(default_threshold)).astype(float)
        frame["y"] = (frame["y_count"] >= threshold_series).astype(int)
    else:
        frame["y"] = (frame["y_count"] >= int(threshold)).astype(int)
    return frame


def fit_region_train_stats(chicago_train_df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        chicago_train_df.groupby("region_id")["lag_0"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "region_train_mean",
                "std": "region_train_std",
                "count": "region_train_count",
            }
        )
    )
    stats["region_train_std"] = stats["region_train_std"].fillna(0.0)
    return stats


def apply_region_train_stats(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.merge(stats, on="region_id", how="left", indicator=True)
    frame["region_seen_in_train"] = (frame["_merge"] == "both").astype(int)
    frame = frame.drop(columns=["_merge"])

    defaults = {
        "region_train_mean": float(stats["region_train_mean"].mean()) if not stats.empty else 0.0,
        "region_train_std": float(stats["region_train_std"].mean()) if not stats.empty else 0.0,
        "region_train_count": float(stats["region_train_count"].mean()) if not stats.empty else 0.0,
    }
    for col, val in defaults.items():
        frame[col] = frame[col].fillna(val)

    return frame


def enforce_xy_schema(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    dataset_name: str,
) -> pd.DataFrame:
    required_front = ["region_id", "time_id"]
    required_back = ["y", "y_count"]
    cols = required_front + list(feature_columns) + required_back

    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset {dataset_name} missing columns: {missing}")

    out = frame[cols].copy()
    out["region_id"] = out["region_id"].astype(str)
    out["time_id"] = pd.to_datetime(out["time_id"]).dt.strftime("%Y-%m-%d")
    out["y"] = out["y"].astype(int)
    out["y_count"] = out["y_count"].astype(float)

    return out


def check_schema_consistency(frames: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    keys = list(frames.keys())
    base_cols = list(frames[keys[0]].columns)
    consistent = True
    mismatches: Dict[str, List[str]] = {}

    for name, frame in frames.items():
        cols = list(frame.columns)
        if cols != base_cols:
            consistent = False
            mismatches[name] = cols

    return {
        "consistent": consistent,
        "base_dataset": keys[0],
        "base_columns": base_cols,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone weekly feature generation utility.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet/csv with columns event_date, region_id, crime_group")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time_window_days", type=int, default=7, choices=[3, 7])
    parser.add_argument("--anchor_date", default="2015-01-01")
    args = parser.parse_args()

    if args.input.suffix.lower() == ".parquet":
        events = pd.read_parquet(args.input)
    else:
        events = pd.read_csv(args.input)

    panel = aggregate_weekly_events(
        events,
        time_window_days=args.time_window_days,
        anchor_date=args.anchor_date,
    )
    sup = build_supervised_weekly(panel, time_window_days=args.time_window_days)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sup.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
