from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .features import make_time_id

MODEL_ORDER = [
    "baseline_persistence",
    "logistic_regression",
    "random_forest",
    "lstm",
]
DOMAIN_ORDER = ["chicago_2025", "nibrs"]
CRIME_GROUP_ORDER = [
    "theft_larceny",
    "assault_battery",
    "burglary",
    "robbery",
    "motor_vehicle_theft",
    "drug_narcotics",
]
CHICAGO_LON_RANGE = (-88.0, -87.5)
CHICAGO_LAT_RANGE = (41.5, 42.1)


def _clean_region_token(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("UNKNOWN").astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    return cleaned.str.upper()


def _extract_state_from_region(region_id: object) -> str:
    token = str(region_id).strip().upper()
    if "__" in token:
        token = token.split("__", 1)[0]
    return token if token else "UNK"


def _build_state_neighbor_map(region_ids: Sequence[object]) -> Dict[str, Set[str]]:
    by_state: Dict[str, List[str]] = {}
    for rid in region_ids:
        rid_s = str(rid)
        state = _extract_state_from_region(rid_s)
        by_state.setdefault(state, []).append(rid_s)

    neighbor_map: Dict[str, Set[str]] = {}
    for _, regions in by_state.items():
        uniq = sorted(set(regions))
        region_set = set(uniq)
        for rid in uniq:
            neighbor_map[rid] = region_set - {rid}
    return neighbor_map


def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def compute_overall_metrics_by_seed(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (seed, domain, model), g in predictions.groupby(["seed", "dataset", "model"]):
        y_true = g["y"].astype(int)
        y_pred = g["y_pred"].astype(int)
        y_score = g["y_score"].astype(float)

        rows.append(
            {
                "seed": int(seed),
                "domain": str(domain),
                "model": str(model),
                "auc": _safe_auc(y_true, y_score),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "n_rows": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def compute_temporal_metrics_by_seed(predictions: pd.DataFrame, period: str = "M") -> pd.DataFrame:
    frame = predictions.copy()
    frame["time_id"] = pd.to_datetime(frame["time_id"], errors="coerce")
    frame["date_bucket"] = frame["time_id"].dt.to_period(period).astype(str)

    rows: List[Dict[str, object]] = []
    for (seed, domain, model, bucket), g in frame.groupby(["seed", "dataset", "model", "date_bucket"]):
        y_true = g["y"].astype(int)
        y_pred = g["y_pred"].astype(int)
        rows.append(
            {
                "seed": int(seed),
                "domain": str(domain),
                "model": str(model),
                "date_bucket": str(bucket),
                "metric": "f1",
                "value": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def compute_neighbor_metrics_by_seed(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (seed, domain, model), g in predictions.groupby(["seed", "dataset", "model"]):
        frame = g.copy().reset_index(drop=True)
        frame["region_id"] = frame["region_id"].astype(str)
        frame["time_id"] = pd.to_datetime(frame["time_id"], errors="coerce")
        frame = frame.sort_values(["time_id", "region_id"]).reset_index(drop=True)
        if frame.empty:
            continue

        neighbor_map = _build_state_neighbor_map(frame["region_id"].unique().tolist())
        neighbor_true = np.zeros(len(frame), dtype=np.int64)

        for _, time_block in frame.groupby("time_id", sort=False):
            true_regions = set(time_block.loc[time_block["y"].astype(int) == 1, "region_id"].astype(str).tolist())
            if not true_regions:
                continue

            for row_idx in time_block.index.tolist():
                rid = str(frame.at[row_idx, "region_id"])
                if rid in true_regions:
                    neighbor_true[row_idx] = 1
                    continue
                if neighbor_map.get(rid, set()) & true_regions:
                    neighbor_true[row_idx] = 1

        y_pred = frame["y_pred"].astype(int).to_numpy()
        y_score = frame["y_score"].astype(float).to_numpy()
        y_neighbor = pd.Series(neighbor_true.astype(int))

        rows.append(
            {
                "seed": int(seed),
                "domain": str(domain),
                "model": str(model),
                "neighbor_metric": "auc",
                "value": _safe_auc(y_neighbor, pd.Series(y_score)),
            }
        )
        rows.append(
            {
                "seed": int(seed),
                "domain": str(domain),
                "model": str(model),
                "neighbor_metric": "f1",
                "value": float(f1_score(y_neighbor, y_pred, zero_division=0)),
            }
        )
        rows.append(
            {
                "seed": int(seed),
                "domain": str(domain),
                "model": str(model),
                "neighbor_metric": "recall",
                "value": float(recall_score(y_neighbor, y_pred, zero_division=0)),
            }
        )
        rows.append(
            {
                "seed": int(seed),
                "domain": str(domain),
                "model": str(model),
                "neighbor_metric": "precision",
                "value": float(precision_score(y_neighbor, y_pred, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def compute_crimegroup_metrics_by_seed(predictions: pd.DataFrame) -> pd.DataFrame:
    if "crime_group" not in predictions.columns:
        return pd.DataFrame(columns=["seed", "domain", "model", "crime_group", "metric", "value"])

    subset = predictions[predictions["dataset"] == "nibrs"].copy()
    if subset.empty or subset["crime_group"].nunique() <= 1:
        return pd.DataFrame(columns=["seed", "domain", "model", "crime_group", "metric", "value"])

    rows: List[Dict[str, object]] = []

    for (seed, model, crime_group), g in subset.groupby(["seed", "model", "crime_group"]):
        y_true = g["y"].astype(int)
        y_pred = g["y_pred"].astype(int)
        y_score = g["y_score"].astype(float)

        rows.append(
            {
                "seed": int(seed),
                "domain": "nibrs",
                "model": str(model),
                "crime_group": str(crime_group),
                "metric": "auc",
                "value": _safe_auc(y_true, y_score),
            }
        )
        rows.append(
            {
                "seed": int(seed),
                "domain": "nibrs",
                "model": str(model),
                "crime_group": str(crime_group),
                "metric": "f1",
                "value": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def build_metrics_long(predictions: pd.DataFrame, time_window_days: int) -> pd.DataFrame:
    overall = compute_overall_metrics_by_seed(predictions)

    overall_long_rows: List[Dict[str, object]] = []
    for _, row in overall.iterrows():
        for metric in ["auc", "f1", "recall", "precision"]:
            overall_long_rows.append(
                {
                    "time_window_days": int(time_window_days),
                    "domain": row["domain"],
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "metric": metric,
                    "value": float(row[metric]),
                    "crime_group": "",
                    "date_bucket": "",
                    "slice": "overall",
                }
            )

    temporal = compute_temporal_metrics_by_seed(predictions, period="M")
    temporal_rows: List[Dict[str, object]] = []
    for _, row in temporal.iterrows():
        temporal_rows.append(
            {
                "time_window_days": int(time_window_days),
                "domain": row["domain"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "metric": row["metric"],
                "value": float(row["value"]),
                "crime_group": "",
                "date_bucket": row["date_bucket"],
                "slice": "temporal",
            }
        )

    cg = compute_crimegroup_metrics_by_seed(predictions)
    cg_rows: List[Dict[str, object]] = []
    for _, row in cg.iterrows():
        cg_rows.append(
            {
                "time_window_days": int(time_window_days),
                "domain": row["domain"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "metric": row["metric"],
                "value": float(row["value"]),
                "crime_group": row["crime_group"],
                "date_bucket": "",
                "slice": "crime_group",
            }
        )

    neighbor = compute_neighbor_metrics_by_seed(predictions)
    neighbor_rows: List[Dict[str, object]] = []
    for _, row in neighbor.iterrows():
        neighbor_rows.append(
            {
                "time_window_days": int(time_window_days),
                "domain": row["domain"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "metric": row["neighbor_metric"],
                "value": float(row["value"]),
                "crime_group": "",
                "date_bucket": "",
                "slice": "neighbor_overall",
            }
        )

    long_df = pd.DataFrame(overall_long_rows + temporal_rows + cg_rows + neighbor_rows)
    return long_df


def check_metrics_long_completeness(
    metrics_long: pd.DataFrame,
    expected_models: Sequence[str],
    expected_seeds: Sequence[int],
) -> Dict[str, object]:
    required_metrics = ["auc", "f1", "recall", "precision"]
    domains = ["chicago_2025", "nibrs"]

    base = metrics_long[(metrics_long["slice"] == "overall") & (metrics_long["metric"].isin(required_metrics))].copy()
    combos = {
        (str(r.model), int(r.seed), str(r.domain), str(r.metric))
        for r in base[["model", "seed", "domain", "metric"]].itertuples(index=False)
    }

    expected = {
        (m, int(s), d, metric)
        for m in expected_models
        for s in expected_seeds
        for d in domains
        for metric in required_metrics
    }

    missing = sorted(expected - combos)
    if missing:
        raise ValueError(
            f"metrics_long completeness check failed. Missing {len(missing)} combinations. Example: {missing[:8]}"
        )

    return {
        "ok": True,
        "n_rows": int(len(metrics_long)),
        "expected_combinations": int(len(expected)),
        "present_combinations": int(len(combos)),
    }


def build_table1_distribution(
    chicago_train_events: pd.DataFrame,
    chicago_test_events: pd.DataFrame,
    nibrs_events: pd.DataFrame,
    chicago_train_xy: pd.DataFrame,
    chicago_2025_xy: pd.DataFrame,
    nibrs_xy: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _event_stats(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        counts = df["crime_group"].value_counts().reindex(CRIME_GROUP_ORDER, fill_value=0)
        pct = counts / max(int(counts.sum()), 1)
        return counts, pct

    def _pos_rate(df: pd.DataFrame) -> pd.Series:
        if "crime_group" not in df.columns:
            overall = float(df["y"].mean()) if not df.empty else 0.0
            return pd.Series({cg: overall for cg in CRIME_GROUP_ORDER})
        return df.groupby("crime_group")["y"].mean().reindex(CRIME_GROUP_ORDER).fillna(0.0)

    ctrain_n, ctrain_pct = _event_stats(chicago_train_events)
    c2025_n, c2025_pct = _event_stats(chicago_test_events)
    nibrs_n, nibrs_pct = _event_stats(nibrs_events)

    ctrain_pos = _pos_rate(chicago_train_xy)
    c2025_pos = _pos_rate(chicago_2025_xy)
    nibrs_pos = _pos_rate(nibrs_xy)

    rows: List[Dict[str, object]] = []
    for cg in CRIME_GROUP_ORDER:
        rows.append(
            {
                "crime_group": cg,
                "Chicago_train_2015_2024_incidents_n": int(ctrain_n[cg]),
                "Chicago_train_2015_2024_incidents_pct": float(ctrain_pct[cg]),
                "Chicago_train_2015_2024_positive_rate": float(ctrain_pos[cg]),
                "Chicago_2025_incidents_n": int(c2025_n[cg]),
                "Chicago_2025_incidents_pct": float(c2025_pct[cg]),
                "Chicago_2025_positive_rate": float(c2025_pos[cg]),
                "NIBRS_incidents_n": int(nibrs_n[cg]),
                "NIBRS_incidents_pct": float(nibrs_pct[cg]),
                "NIBRS_positive_rate": float(nibrs_pos[cg]),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table1_distribution.csv", index=False)
    (out_dir / "table1_distribution.md").write_text(_markdown_table(table), encoding="utf-8")
    return table


def build_table2_results(
    metrics_long: pd.DataFrame,
    out_dir: Path,
    model_order: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    base = metrics_long[(metrics_long["slice"] == "overall") & (metrics_long["metric"].isin(["auc", "f1", "recall", "precision"]))]
    agg = (
        base.groupby(["model", "domain", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .fillna(0.0)
    )
    agg = agg.sort_values(["model", "domain", "metric"]).reset_index(drop=True)

    fmt = []
    for model in model_order:
        row = {"model": model}
        for domain, tag in [("chicago_2025", "Chicago2025"), ("nibrs", "NIBRS")]:
            for metric, mtag in [("auc", "AUC"), ("f1", "F1"), ("recall", "Recall"), ("precision", "Precision")]:
                sub = agg[(agg["model"] == model) & (agg["domain"] == domain) & (agg["metric"] == metric)]
                if sub.empty:
                    cell = "NA"
                else:
                    mean = float(sub["mean"].iloc[0])
                    std = float(sub["std"].iloc[0])
                    cell = f"{mean:.3f} ± {std:.3f}"
                row[f"{tag}_{mtag}"] = cell
        fmt.append(row)

    table_fmt = pd.DataFrame(fmt)
    table_fmt.to_csv(out_dir / "table2_results.csv", index=False)
    (out_dir / "table2_results.md").write_text(_markdown_table(table_fmt), encoding="utf-8")
    agg.to_csv(out_dir / "table2_results_numeric.csv", index=False)
    return table_fmt, agg


def plot_figure1_overall_bars(metrics_long: pd.DataFrame, out_dir: Path, model_order: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["auc", "f1", "precision", "recall"]
    base = metrics_long[(metrics_long["slice"] == "overall") & (metrics_long["metric"].isin(metrics_to_plot))]

    for metric in metrics_to_plot:
        mdf = base[base["metric"] == metric]
        agg = (
            mdf.groupby(["domain", "model"], as_index=False)
            .agg(
                mean=("value", "mean"),
                std=("value", "std"),
            )
            .fillna(0.0)
        )

        x = np.arange(len(model_order), dtype=float)
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, domain in enumerate(DOMAIN_ORDER):
            means = []
            errs = []
            for model in model_order:
                sub = agg[(agg["domain"] == domain) & (agg["model"] == model)]
                means.append(float(sub["mean"].iloc[0]) if not sub.empty else np.nan)
                errs.append(float(sub["std"].iloc[0]) if not sub.empty else 0.0)

            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width=width, yerr=errs, capsize=3, label=domain)

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Figure 1: Overall {metric.upper()} (Chicago2025 vs NIBRS)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(out_dir / f"fig1_overall_{metric}.png", dpi=180)
        plt.close(fig)


def plot_neighboraware_comparison(metrics_long: pd.DataFrame, out_dir: Path, model_order: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    standard = metrics_long[(metrics_long["slice"] == "overall") & (metrics_long["metric"] == "f1")].copy()
    neighbor = metrics_long[(metrics_long["slice"] == "neighbor_overall") & (metrics_long["metric"] == "f1")].copy()
    if standard.empty or neighbor.empty:
        return

    for domain, fname in [
        ("chicago_2025", "fig_neighboraware_f1_chicago2025.png"),
        ("nibrs", "fig_neighboraware_f1_nibrs.png"),
    ]:
        s = (
            standard[standard["domain"] == domain]
            .groupby(["model"], as_index=False)
            .agg(
                mean=("value", "mean"),
                std=("value", "std"),
            )
            .set_index("model")
        )
        n = (
            neighbor[neighbor["domain"] == domain]
            .groupby(["model"], as_index=False)
            .agg(
                mean=("value", "mean"),
                std=("value", "std"),
            )
            .set_index("model")
        )
        if s.empty or n.empty:
            continue

        x = np.arange(len(model_order), dtype=float)
        width = 0.36
        s_m = np.array([float(s.loc[m, "mean"]) if m in s.index else np.nan for m in model_order], dtype=float)
        s_e = np.array([float(s.loc[m, "std"]) if m in s.index else 0.0 for m in model_order], dtype=float)
        n_m = np.array([float(n.loc[m, "mean"]) if m in n.index else np.nan for m in model_order], dtype=float)
        n_e = np.array([float(n.loc[m, "std"]) if m in n.index else 0.0 for m in model_order], dtype=float)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, s_m, yerr=s_e, width=width, capsize=3, label="Exact-region F1")
        ax.bar(x + width / 2, n_m, yerr=n_e, width=width, capsize=3, label="Neighbor-aware F1")
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20)
        ax.set_ylabel("F1")
        ax.set_title(f"Neighbor-aware vs Exact-region F1 ({domain})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)


def plot_figure3_temporal_curves(metrics_long: pd.DataFrame, out_dir: Path, model_order: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    temporal = metrics_long[(metrics_long["slice"] == "temporal") & (metrics_long["metric"] == "f1")].copy()
    if temporal.empty:
        return

    for domain, fname in [
        ("chicago_2025", "fig3a_temporal_chicago2025_f1.png"),
        ("nibrs", "fig3b_temporal_nibrs_f1.png"),
    ]:
        df = temporal[temporal["domain"] == domain].copy()
        if df.empty:
            continue

        periods = sorted(df["date_bucket"].unique().tolist())
        fig, ax = plt.subplots(figsize=(12, 5))

        for model in model_order:
            m = (
                df[df["model"] == model]
                .groupby("date_bucket", as_index=False)
                .agg(
                    mean=("value", "mean"),
                    std=("value", "std"),
                )
                .set_index("date_bucket")
                .reindex(periods)
            )
            y = m["mean"].to_numpy(dtype=float)
            s = m["std"].fillna(0.0).to_numpy(dtype=float)
            x = np.arange(len(periods))
            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, y - s, y + s, alpha=0.15)

        ax.set_xticks(np.arange(len(periods)))
        ax.set_xticklabels(periods, rotation=45, ha="right")
        ax.set_ylabel("F1")
        ax.set_title(f"Figure 3: Temporal Stability ({domain})")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)


def compute_topk_coverage(
    predictions: pd.DataFrame,
    k_candidates: Sequence[int] = (5, 10, 20, 50, 100),
) -> Tuple[pd.DataFrame, Dict[str, List[int]], Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    k_map: Dict[str, List[int]] = {}
    region_n_map: Dict[str, int] = {}

    for domain in DOMAIN_ORDER:
        subset = predictions[predictions["dataset"] == domain].copy()
        if subset.empty:
            continue

        agg = (
            subset.groupby(["seed", "model", "time_id", "region_id"], as_index=False)
            .agg(risk_score=("y_score", "max"), true_incidents=("y_count", "sum"))
        )
        if agg.empty:
            continue

        n_regions = int(agg["region_id"].nunique())
        region_n_map[domain] = n_regions
        k_list = [int(k) for k in k_candidates if int(k) <= n_regions]
        if not k_list:
            k_list = [max(1, n_regions)]
        k_map[domain] = k_list

        for (seed, model, time_id), g in agg.groupby(["seed", "model", "time_id"]):
            g = g.sort_values("risk_score", ascending=False)
            total = float(g["true_incidents"].sum())
            for k in k_list:
                top = g.head(k)
                top_sum = float(top["true_incidents"].sum())
                coverage = top_sum / total if total > 0 else np.nan
                rows.append(
                    {
                        "domain": domain,
                        "seed": int(seed),
                        "model": str(model),
                        "time_id": str(time_id),
                        "k": int(k),
                        "coverage": float(coverage),
                    }
                )

    cover_df = pd.DataFrame(rows)
    if cover_df.empty:
        return pd.DataFrame(columns=["domain", "seed", "model", "k", "coverage"]), k_map, region_n_map
    by_seed = cover_df.groupby(["domain", "seed", "model", "k"], as_index=False)["coverage"].mean()
    return by_seed, k_map, region_n_map


def plot_figure4_topk_coverage(topk_by_seed: pd.DataFrame, out_dir: Path, model_order: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if topk_by_seed.empty:
        return

    for domain, fname in [
        ("chicago_2025", "fig4_topk_coverage_chicago2025.png"),
        ("nibrs", "fig4_topk_coverage_nibrs.png"),
    ]:
        sub = topk_by_seed[topk_by_seed["domain"] == domain].copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby(["model", "k"], as_index=False)
            .agg(
                mean=("coverage", "mean"),
                std=("coverage", "std"),
            )
            .fillna(0.0)
        )
        ks = sorted(agg["k"].unique().tolist())

        fig, ax = plt.subplots(figsize=(8, 5))
        for model in model_order:
            m = agg[agg["model"] == model].set_index("k").reindex(ks)
            y = m["mean"].to_numpy(dtype=float)
            s = m["std"].fillna(0.0).to_numpy(dtype=float)
            x = np.array(ks, dtype=float)
            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, y - s, y + s, alpha=0.12)

        ax.set_xlabel("K")
        ax.set_ylabel("Coverage@K")
        ax.set_title(f"Figure 4: Top-K Deployment Curve on {domain}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points

    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts

    pts_list = sorted((float(x), float(y)) for x, y in pts.tolist())

    def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts_list:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _build_chicago_geometry(chicago_geo_points: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    required = {"region_id", "latitude", "longitude"}
    if chicago_geo_points.empty or not required.issubset(chicago_geo_points.columns):
        return {}, pd.DataFrame(columns=["region_id", "longitude", "latitude", "district_label"])

    working = chicago_geo_points.copy()
    working["region_id"] = working["region_id"].astype(str)
    working["latitude"] = pd.to_numeric(working["latitude"], errors="coerce")
    working["longitude"] = pd.to_numeric(working["longitude"], errors="coerce")
    working = working.dropna(subset=["latitude", "longitude", "region_id"])
    if working.empty:
        return {}, pd.DataFrame(columns=["region_id", "longitude", "latitude", "district_label"])

    rng = np.random.default_rng(0)
    polygons: Dict[str, np.ndarray] = {}
    centroid_rows: List[Dict[str, object]] = []

    for region, g in working.groupby("region_id"):
        pts = g[["longitude", "latitude"]].to_numpy(dtype=float)
        if pts.shape[0] == 0:
            continue
        if pts.shape[0] > 3000:
            idx = rng.choice(pts.shape[0], size=3000, replace=False)
            pts = pts[idx]
        hull = _convex_hull(pts)
        polygons[str(region)] = hull

        centroid_rows.append(
            {
                "region_id": str(region),
                "longitude": float(np.median(pts[:, 0])),
                "latitude": float(np.median(pts[:, 1])),
                "district_label": str(region).split("CPD_DIST_")[-1],
            }
        )

    centroids = pd.DataFrame(centroid_rows)
    return polygons, centroids


def _plot_region_choropleth(
    ax,
    polygons: Dict[str, np.ndarray],
    centroids: pd.DataFrame,
    values: Mapping[str, float],
    title: str,
    cmap_name: str = "YlOrRd",
) -> None:
    from matplotlib.patches import Polygon as MplPolygon

    if centroids.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No geometry available", ha="center", va="center")
        return

    regions = centroids["region_id"].astype(str).tolist()
    vals = np.array([values.get(r, np.nan) for r in regions], dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    centroid_lookup = centroids.set_index("region_id")[["longitude", "latitude", "district_label"]].to_dict("index")
    for region in regions:
        val = values.get(region, np.nan)
        color = "#E0E0E0" if not np.isfinite(val) else cmap(norm(float(val)))
        hull = polygons.get(region)
        if hull is not None and hull.shape[0] >= 3:
            patch = MplPolygon(
                hull,
                closed=True,
                facecolor=color,
                edgecolor="#555555",
                linewidth=0.7,
                alpha=0.9,
            )
            ax.add_patch(patch)
        else:
            c = centroid_lookup[region]
            ax.scatter(c["longitude"], c["latitude"], s=120, c=[color], edgecolor="#555555", linewidth=0.7)

    for region in regions:
        c = centroid_lookup[region]
        ax.text(
            float(c["longitude"]),
            float(c["latitude"]),
            str(c["district_label"]),
            ha="center",
            va="center",
            fontsize=7,
            color="#1A1A1A",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def compute_chicago_topk_selection_frequency(
    predictions: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    subset = predictions[predictions["dataset"] == "chicago_2025"].copy()
    if subset.empty:
        return pd.DataFrame(columns=["seed", "model", "region_id", "topk_selection_rate"])

    rows: List[Dict[str, object]] = []
    for (seed, model), g_sm in subset.groupby(["seed", "model"]):
        windows = sorted(g_sm["time_id"].astype(str).unique().tolist())
        if not windows:
            continue
        per_region_count: Dict[str, int] = {}
        for time_id, g in g_sm.groupby("time_id"):
            g = g.sort_values("y_score", ascending=False)
            kk = max(1, min(int(k), int(g["region_id"].nunique())))
            selected = set(g.head(kk)["region_id"].astype(str).tolist())
            for region in g["region_id"].astype(str).unique().tolist():
                per_region_count.setdefault(region, 0)
                if region in selected:
                    per_region_count[region] += 1
        total_windows = len(windows)
        for region, count in per_region_count.items():
            rows.append(
                {
                    "seed": int(seed),
                    "model": str(model),
                    "region_id": str(region),
                    "topk_selection_rate": float(count / max(total_windows, 1)),
                }
            )

    by_seed = pd.DataFrame(rows)
    if by_seed.empty:
        return by_seed
    return by_seed


def compute_chicago_event_hit_classes(
    predictions: pd.DataFrame,
    chicago_geo_events: pd.DataFrame,
    time_window_days: int,
    anchor_date: str,
    model: str,
    seed: int,
    geo_neighbors: Mapping[str, Set[str]],
    k: int = 5,
) -> pd.DataFrame:
    if predictions.empty or chicago_geo_events.empty:
        return pd.DataFrame(
            columns=[
                "region_id",
                "event_date",
                "latitude",
                "longitude",
                "event_time_id",
                "pred_time_id",
                "exact_hit",
                "neighbor_hit",
                "coverage_class",
            ]
        )

    pred = predictions[
        (predictions["dataset"] == "chicago_2025")
        & (predictions["model"] == model)
        & (predictions["seed"] == int(seed))
    ].copy()
    if pred.empty:
        return pd.DataFrame(
            columns=[
                "region_id",
                "event_date",
                "latitude",
                "longitude",
                "event_time_id",
                "pred_time_id",
                "exact_hit",
                "neighbor_hit",
                "coverage_class",
            ]
        )

    pred["time_id"] = pd.to_datetime(pred["time_id"], errors="coerce")
    topk_by_time: Dict[pd.Timestamp, Set[str]] = {}
    for time_id, g in pred.groupby("time_id"):
        kk = max(1, min(int(k), int(g["region_id"].nunique())))
        selected = set(g.sort_values("y_score", ascending=False).head(kk)["region_id"].astype(str).tolist())
        topk_by_time[pd.Timestamp(time_id)] = selected

    ev = chicago_geo_events.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev["region_id"] = ev["region_id"].astype(str)
    ev["latitude"] = pd.to_numeric(ev["latitude"], errors="coerce")
    ev["longitude"] = pd.to_numeric(ev["longitude"], errors="coerce")
    ev = ev.dropna(subset=["event_date", "region_id", "latitude", "longitude"]).copy()
    if ev.empty:
        return ev

    ev["event_time_id"] = make_time_id(
        ev["event_date"],
        time_window_days=int(time_window_days),
        anchor_date=str(anchor_date),
    )
    ev["pred_time_id"] = pd.to_datetime(ev["event_time_id"], errors="coerce") - pd.Timedelta(days=int(time_window_days))

    exact_hits: List[bool] = []
    neighbor_hits: List[bool] = []
    for _, row in ev.iterrows():
        region = str(row["region_id"])
        pred_time = pd.Timestamp(row["pred_time_id"])
        selected = topk_by_time.get(pred_time, set())
        exact = region in selected
        neigh = (not exact) and (len(geo_neighbors.get(region, set()) & selected) > 0)
        exact_hits.append(bool(exact))
        neighbor_hits.append(bool(neigh))

    ev["exact_hit"] = np.array(exact_hits, dtype=bool)
    ev["neighbor_hit"] = np.array(neighbor_hits, dtype=bool)
    ev["coverage_class"] = np.where(
        ev["exact_hit"],
        "Exact hit",
        np.where(ev["neighbor_hit"], "Neighbor hit", "Miss"),
    )
    return ev


def _plot_chicago_boundaries(
    ax,
    polygons: Dict[str, np.ndarray],
    centroids: pd.DataFrame,
    lon_range: Tuple[float, float] = CHICAGO_LON_RANGE,
    lat_range: Tuple[float, float] = CHICAGO_LAT_RANGE,
) -> None:
    from matplotlib.patches import Polygon as MplPolygon

    for region, hull in polygons.items():
        if hull is None or hull.shape[0] < 3:
            continue
        patch = MplPolygon(
            hull,
            closed=True,
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.add_patch(patch)

    if not centroids.empty:
        for _, row in centroids.iterrows():
            ax.text(
                float(row["longitude"]),
                float(row["latitude"]),
                str(row.get("district_label", "")),
                ha="center",
                va="center",
                fontsize=6,
                color="#333333",
                alpha=0.9,
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(lon_range[0]), float(lon_range[1]))
    ax.set_ylim(float(lat_range[0]), float(lat_range[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def plot_geo_chicago_event_hitmaps(
    predictions: pd.DataFrame,
    metrics_long: pd.DataFrame,
    chicago_geo_points: pd.DataFrame,
    chicago_geo_events: pd.DataFrame,
    time_window_days: int,
    anchor_date: str,
    out_dir: Path,
    model_order: Sequence[str],
    topk_k: int = 5,
    sample_n: int = 50000,
) -> Tuple[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    polygons, centroids = _build_chicago_geometry(chicago_geo_points)
    if centroids.empty or chicago_geo_events.empty:
        return "", pd.DataFrame()

    best_model = _select_best_chicago_model(metrics_long, model_order=model_order)
    seed_values = predictions["seed"].dropna().astype(int).tolist()
    chosen_seed = int(min(seed_values)) if seed_values else 0
    geo_neighbors = _build_knn_neighbors(centroids, k=3)

    event_hits = compute_chicago_event_hit_classes(
        predictions=predictions,
        chicago_geo_events=chicago_geo_events,
        time_window_days=int(time_window_days),
        anchor_date=str(anchor_date),
        model=str(best_model),
        seed=chosen_seed,
        geo_neighbors=geo_neighbors,
        k=int(topk_k),
    )
    if event_hits.empty:
        return str(best_model), event_hits

    if len(event_hits) > int(sample_n):
        # Preserve class proportions while reducing drawing load.
        frac = float(sample_n) / float(len(event_hits))
        sampled = (
            event_hits.groupby("coverage_class", group_keys=False)
            .apply(lambda g: g.sample(max(1, int(np.floor(len(g) * frac))), random_state=42))
            .reset_index(drop=True)
        )
        if len(sampled) > int(sample_n):
            sampled = sampled.sample(int(sample_n), random_state=42).reset_index(drop=True)
    else:
        sampled = event_hits.copy()

    if "coverage_class" not in sampled.columns:
        sampled["coverage_class"] = np.where(
            sampled["exact_hit"],
            "Exact hit",
            np.where(sampled["neighbor_hit"], "Neighbor hit", "Miss"),
        )

    exact_rate = float(event_hits["exact_hit"].mean())
    neighbor_rate = float((event_hits["exact_hit"] | event_hits["neighbor_hit"]).mean())

    # Figure A: exact hit vs miss.
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    left = sampled.copy()
    left["label"] = np.where(left["exact_hit"], "Exact hit", "Miss")
    palette_left = {"Exact hit": "#2E8B57", "Miss": "#D1495B"}
    for label, color in palette_left.items():
        sub = left[left["label"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=8,
            c=color,
            alpha=0.45,
            label=label,
            linewidths=0,
        )
    _plot_chicago_boundaries(ax, polygons=polygons, centroids=centroids)
    ax.set_title(f"Chicago Exact-region Coverage ({best_model}, top-{topk_k})")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_geo_chicago_event_hitmap_exact.png", dpi=180)
    plt.close(fig)

    # Figure B: exact + neighbor hits vs miss.
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    right = sampled.copy()
    palette_right = {"Exact hit": "#2E8B57", "Neighbor hit": "#F4A259", "Miss": "#D1495B"}
    for label, color in palette_right.items():
        sub = right[right["coverage_class"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=8,
            c=color,
            alpha=0.45,
            label=label,
            linewidths=0,
        )
    _plot_chicago_boundaries(ax, polygons=polygons, centroids=centroids)
    ax.set_title(
        f"Chicago Neighbor-aware Coverage ({best_model}, top-{topk_k})\n"
        f"Exact={exact_rate:.3f}, Neighbor-aware={neighbor_rate:.3f}"
    )
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_geo_chicago_event_hitmap_neighbor.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "time_window_days": int(time_window_days),
                "model": str(best_model),
                "seed": int(chosen_seed),
                "topk_k": int(topk_k),
                "n_events_total": int(len(event_hits)),
                "n_events_sampled_for_plot": int(len(sampled)),
                "exact_hit_rate": exact_rate,
                "neighbor_aware_hit_rate": neighbor_rate,
            }
        ]
    )
    return str(best_model), summary


def plot_geo_chicago_topk_frequency(
    predictions: pd.DataFrame,
    chicago_geo_points: pd.DataFrame,
    out_dir: Path,
    model_order: Sequence[str],
    k: int = 5,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_seed = compute_chicago_topk_selection_frequency(predictions, k=k)
    if by_seed.empty or chicago_geo_points.empty:
        return by_seed

    agg = by_seed.groupby(["model", "region_id"], as_index=False)["topk_selection_rate"].mean()
    polygons, centroids = _build_chicago_geometry(chicago_geo_points)
    if centroids.empty:
        return by_seed

    models = [m for m in model_order if m in agg["model"].unique().tolist()]
    n_models = len(models)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()

    for idx, model in enumerate(models):
        ax = axes_arr[idx]
        val_map = (
            agg[agg["model"] == model]
            .set_index("region_id")["topk_selection_rate"]
            .to_dict()
        )
        _plot_region_choropleth(
            ax,
            polygons=polygons,
            centroids=centroids,
            values=val_map,
            title=f"{model} | Top-{k} selection frequency",
            cmap_name="YlOrRd",
        )

    for idx in range(n_models, len(axes_arr)):
        axes_arr[idx].axis("off")

    fig.suptitle("Chicago Geo Map: Top-K Region Selection Frequency", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "fig_geo_chicago_topk_selection_frequency.png", dpi=180)
    plt.close(fig)
    return by_seed


def _build_knn_neighbors(centroids: pd.DataFrame, k: int = 3) -> Dict[str, Set[str]]:
    if centroids.empty:
        return {}
    rid = centroids["region_id"].astype(str).to_numpy()
    xy = centroids[["longitude", "latitude"]].to_numpy(dtype=float)
    neighbors: Dict[str, Set[str]] = {}
    for i in range(len(rid)):
        d = np.sqrt(((xy - xy[i]) ** 2).sum(axis=1))
        order = np.argsort(d)
        chosen = [rid[j] for j in order if j != i][: max(1, min(k, len(rid) - 1))]
        neighbors[str(rid[i])] = set(chosen)
    return neighbors


def _select_best_chicago_model(metrics_long: pd.DataFrame, model_order: Sequence[str]) -> str:
    base = metrics_long[
        (metrics_long["slice"] == "overall")
        & (metrics_long["domain"] == "chicago_2025")
        & (metrics_long["metric"] == "f1")
    ].copy()
    if base.empty:
        return model_order[0]
    ranked = base.groupby("model", as_index=False)["value"].mean().sort_values("value", ascending=False)
    for m in ranked["model"].tolist():
        if m in model_order:
            return str(m)
    return model_order[0]


def compute_chicago_exact_neighbor_rates(
    predictions: pd.DataFrame,
    model: str,
    geo_neighbors: Mapping[str, Set[str]],
) -> pd.DataFrame:
    subset = predictions[(predictions["dataset"] == "chicago_2025") & (predictions["model"] == model)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["region_id", "exact_hit_rate", "neighbor_only_hit_rate", "pred_positive_rate"])

    counts: Dict[str, Dict[str, float]] = {}
    for (seed, time_id), g in subset.groupby(["seed", "time_id"]):
        true_pos = set(g.loc[g["y"].astype(int) == 1, "region_id"].astype(str).tolist())
        for _, row in g.iterrows():
            region = str(row["region_id"])
            pred_pos = int(row["y_pred"]) == 1
            exact_hit = pred_pos and int(row["y"]) == 1
            neighbor_hit = False
            if pred_pos and not exact_hit:
                neighbor_hit = len(geo_neighbors.get(region, set()) & true_pos) > 0

            counts.setdefault(
                region,
                {"n": 0.0, "pred_pos": 0.0, "exact_hit": 0.0, "neighbor_only_hit": 0.0},
            )
            counts[region]["n"] += 1.0
            counts[region]["pred_pos"] += float(pred_pos)
            counts[region]["exact_hit"] += float(exact_hit)
            counts[region]["neighbor_only_hit"] += float(neighbor_hit)

    rows: List[Dict[str, object]] = []
    for region, c in counts.items():
        denom = max(c["n"], 1.0)
        rows.append(
            {
                "region_id": region,
                "exact_hit_rate": float(c["exact_hit"] / denom),
                "neighbor_only_hit_rate": float(c["neighbor_only_hit"] / denom),
                "pred_positive_rate": float(c["pred_pos"] / denom),
            }
        )
    return pd.DataFrame(rows)


def plot_geo_chicago_neighbor_exact(
    predictions: pd.DataFrame,
    metrics_long: pd.DataFrame,
    chicago_geo_points: pd.DataFrame,
    out_dir: Path,
    model_order: Sequence[str],
) -> Tuple[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    polygons, centroids = _build_chicago_geometry(chicago_geo_points)
    if centroids.empty:
        return "", pd.DataFrame()

    best_model = _select_best_chicago_model(metrics_long, model_order=model_order)
    neighbors = _build_knn_neighbors(centroids, k=3)
    rates = compute_chicago_exact_neighbor_rates(predictions, model=best_model, geo_neighbors=neighbors)
    if rates.empty:
        return best_model, rates

    val_exact = rates.set_index("region_id")["exact_hit_rate"].to_dict()
    val_neighbor_only = rates.set_index("region_id")["neighbor_only_hit_rate"].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_region_choropleth(
        axes[0],
        polygons=polygons,
        centroids=centroids,
        values=val_exact,
        title=f"{best_model} | Exact TP rate",
        cmap_name="YlGn",
    )
    _plot_region_choropleth(
        axes[1],
        polygons=polygons,
        centroids=centroids,
        values=val_neighbor_only,
        title=f"{best_model} | Neighbor-only hit rate (kNN)",
        cmap_name="OrRd",
    )
    fig.suptitle("Chicago Geo Map: Exact vs Neighbor-only Hits", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig_geo_chicago_neighbor_vs_exact.png", dpi=180)
    plt.close(fig)
    return best_model, rates


def plot_figure6_heatmap(
    region_bucket_metrics: pd.DataFrame,
    bucket_order: Sequence[str],
    out_dir: Path,
    model_order: Sequence[str],
    metric: str = "auc",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "fig6_heatmap_nibrs_auc.png"
    if region_bucket_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Figure 6 omitted: region-bucket metrics unavailable.",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    mat = (
        region_bucket_metrics.groupby(["model", "region_bucket"], as_index=False)[metric]
        .mean()
        .pivot(index="model", columns="region_bucket", values=metric)
        .reindex(index=model_order, columns=list(bucket_order))
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(bucket_order)))
    ax.set_xticklabels(list(bucket_order), rotation=0)
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.set_title(f"Figure 6: Model x Region-Bucket Heatmap (NIBRS {metric.upper()})")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            label = "NA" if pd.isna(val) else f"{val:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_region_bucket_metrics_by_seed(
    predictions: pd.DataFrame,
    n_buckets: int = 4,
) -> Tuple[pd.DataFrame, List[str], int]:
    subset = predictions[predictions["dataset"] == "nibrs"].copy()
    if subset.empty:
        return pd.DataFrame(), [], 0

    base = subset.drop_duplicates(subset=["region_id", "time_id", "y", "y_count"]).copy()
    if base.empty:
        return pd.DataFrame(), [], 0

    region_workload = base.groupby("region_id")["y_count"].mean().sort_values()
    n_regions = int(region_workload.shape[0])
    if n_regions == 0:
        return pd.DataFrame(), [], 0

    n_bins = max(1, min(int(n_buckets), n_regions))
    if n_bins == 1:
        bucket_labels = ["Q1"]
        region_bucket_map = pd.DataFrame(
            {
                "region_id": region_workload.index.astype(str).to_numpy(),
                "region_bucket": "Q1",
            }
        )
    else:
        bucket_labels = [f"Q{i + 1}" for i in range(n_bins)]
        # Use rank before qcut to avoid duplicate-edge failures on tied values.
        ranks = region_workload.rank(method="first")
        bucket = pd.qcut(ranks, q=n_bins, labels=bucket_labels)
        region_bucket_map = pd.DataFrame(
            {
                "region_id": region_workload.index.astype(str).to_numpy(),
                "region_bucket": bucket.astype(str).to_numpy(),
            }
        )
    region_bucket_map = region_bucket_map.reset_index(drop=True)
    region_bucket_map.index.name = None

    scored = subset.copy()
    scored["region_id"] = scored["region_id"].astype(str)
    scored = scored.merge(region_bucket_map, on="region_id", how="inner")

    rows: List[Dict[str, object]] = []
    for (seed, model, region_bucket), g in scored.groupby(["seed", "model", "region_bucket"]):
        y_true = g["y"].astype(int)
        y_pred = g["y_pred"].astype(int)
        y_score = g["y_score"].astype(float)
        rows.append(
            {
                "seed": int(seed),
                "model": str(model),
                "region_bucket": str(region_bucket),
                "auc": _safe_auc(y_true, y_score),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "n_rows": int(len(g)),
            }
        )

    return pd.DataFrame(rows), bucket_labels, n_regions


def plot_figure6_region_bucket_heatmap(
    region_bucket_metrics: pd.DataFrame,
    bucket_order: Sequence[str],
    out_dir: Path,
    model_order: Sequence[str],
    metric: str = "auc",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"fig6b_heatmap_nibrs_region_bucket_{metric}.png"

    if region_bucket_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Figure 6B omitted: region-bucket metrics unavailable.",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    mat = (
        region_bucket_metrics.groupby(["model", "region_bucket"], as_index=False)[metric]
        .mean()
        .pivot(index="model", columns="region_bucket", values=metric)
        .reindex(index=model_order, columns=list(bucket_order))
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(bucket_order)))
    ax.set_xticklabels(list(bucket_order), rotation=0)
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.set_title(f"Figure 6B: Model x Region-Bucket Heatmap (NIBRS {metric.upper()})")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            label = "NA" if pd.isna(val) else f"{val:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_processed_crime_group_counts(
    chicago_train_events: pd.DataFrame,
    chicago_test_events: pd.DataFrame,
    nibrs_events: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use processed event counts so this figure remains valid for both overall-risk and per-crime tasks.
    chi_train = chicago_train_events["crime_group"].value_counts().reindex(CRIME_GROUP_ORDER, fill_value=0.0)
    chi_2025 = chicago_test_events["crime_group"].value_counts().reindex(CRIME_GROUP_ORDER, fill_value=0.0)
    nibrs = nibrs_events["crime_group"].value_counts().reindex(CRIME_GROUP_ORDER, fill_value=0.0)

    x = np.arange(len(CRIME_GROUP_ORDER), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.38
    bars_train = ax.bar(x - width / 2, chi_train.to_numpy(dtype=float), width=width, label="Chicago train (2015-2024)")
    bars_test = ax.bar(x + width / 2, chi_2025.to_numpy(dtype=float), width=width, label="Chicago 2025")

    ax.set_xticks(x)
    ax.set_xticklabels(CRIME_GROUP_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Incident Count")
    ax.set_title("Crime Group Distribution (Chicago Processed Events)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    for bars in [bars_train, bars_test]:
        for b in bars:
            h = float(b.get_height())
            ax.text(b.get_x() + b.get_width() / 2.0, h, f"{int(round(h)):,}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_processed_counts_chicago.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, nibrs.to_numpy(dtype=float), width=0.6, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(CRIME_GROUP_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Incident Count")
    ax.set_title("Crime Group Distribution (NIBRS Processed Events)")
    ax.grid(axis="y", alpha=0.3)

    for b in bars:
        h = float(b.get_height())
        ax.text(b.get_x() + b.get_width() / 2.0, h, f"{int(round(h)):,}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_processed_counts_nibrs.png", dpi=180)
    plt.close(fig)


def summarize_feature_importance(feature_importance_by_seed: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "model",
        "feature",
        "method",
        "importance_norm_mean",
        "importance_norm_std",
        "importance_raw_mean",
        "importance_raw_std",
        "seed_n",
        "rank_in_model",
    ]
    if feature_importance_by_seed.empty:
        return pd.DataFrame(columns=columns)

    required = {"dataset", "model", "feature", "method", "seed", "importance_norm", "importance_raw"}
    if not required.issubset(feature_importance_by_seed.columns):
        return pd.DataFrame(columns=columns)

    agg = (
        feature_importance_by_seed.groupby(["dataset", "model", "feature", "method"], as_index=False)
        .agg(
            importance_norm_mean=("importance_norm", "mean"),
            importance_norm_std=("importance_norm", "std"),
            importance_raw_mean=("importance_raw", "mean"),
            importance_raw_std=("importance_raw", "std"),
            seed_n=("seed", "nunique"),
        )
        .fillna(0.0)
    )
    agg["rank_in_model"] = (
        agg.groupby(["dataset", "model"])["importance_norm_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    agg = agg.sort_values(["dataset", "model", "rank_in_model"]).reset_index(drop=True)
    return agg[columns]


def plot_feature_importance(
    feature_importance_summary: pd.DataFrame,
    out_dir: Path,
    model_order: Sequence[str],
    dataset: str = "chicago_2025",
    top_n: int = 12,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_name = f"fig_feature_importance_{dataset}.png"
    output_path = out_dir / fig_name

    if feature_importance_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Feature importance unavailable.", ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return fig_name

    focus = feature_importance_summary[feature_importance_summary["dataset"] == dataset].copy()
    if focus.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No feature importance rows for {dataset}.", ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return fig_name

    models = [
        m
        for m in model_order
        if m in focus["model"].unique().tolist() and m != "baseline_persistence"
    ]
    if not models:
        models = [m for m in model_order if m in focus["model"].unique().tolist()]
    if not models:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No matching models for feature importance plot.", ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return fig_name

    n_models = len(models)
    n_cols = 2 if n_models > 1 else 1
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()

    for idx, model in enumerate(models):
        ax = axes_arr[idx]
        mdf = focus[focus["model"] == model].copy()
        if mdf.empty:
            ax.axis("off")
            continue
        mdf = mdf.sort_values("importance_norm_mean", ascending=False).head(int(top_n)).copy()
        mdf = mdf.iloc[::-1]

        y = np.arange(len(mdf))
        x = mdf["importance_norm_mean"].to_numpy(dtype=float)
        s = mdf["importance_norm_std"].fillna(0.0).to_numpy(dtype=float)
        labels = mdf["feature"].astype(str).tolist()

        ax.barh(y, x, xerr=s, color="#4C78A8", alpha=0.9, ecolor="#1F1F1F", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Normalized Importance")
        ax.set_title(f"{model} (top {min(int(top_n), len(mdf))})")
        ax.grid(axis="x", alpha=0.3)

    for idx in range(n_models, len(axes_arr)):
        axes_arr[idx].axis("off")

    fig.suptitle(f"Feature Importance on {dataset} (mean ± std across seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return fig_name


def generate_window_tables_and_figures(
    *,
    time_window_days: int,
    anchor_date: str,
    predictions: pd.DataFrame,
    metrics_long: pd.DataFrame,
    chicago_train_events: pd.DataFrame,
    chicago_test_events: pd.DataFrame,
    nibrs_events: pd.DataFrame,
    chicago_train_xy: pd.DataFrame,
    chicago_2025_xy: pd.DataFrame,
    nibrs_xy: pd.DataFrame,
    chicago_geo_points: pd.DataFrame,
    chicago_geo_events: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    metrics_dir: Path,
    feature_importance_by_seed: pd.DataFrame,
    selected_models: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, object]:
    check = check_metrics_long_completeness(metrics_long, expected_models=selected_models, expected_seeds=seeds)
    (metrics_dir / "metrics_long_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")

    table1 = build_table1_distribution(
        chicago_train_events=chicago_train_events,
        chicago_test_events=chicago_test_events,
        nibrs_events=nibrs_events,
        chicago_train_xy=chicago_train_xy,
        chicago_2025_xy=chicago_2025_xy,
        nibrs_xy=nibrs_xy,
        out_dir=tables_dir,
    )

    table2_fmt, table2_num = build_table2_results(
        metrics_long=metrics_long,
        out_dir=tables_dir,
        model_order=[m for m in MODEL_ORDER if m in selected_models],
    )

    model_order = [m for m in MODEL_ORDER if m in selected_models]
    plot_figure1_overall_bars(metrics_long, figures_dir, model_order=model_order)
    plot_neighboraware_comparison(metrics_long, figures_dir, model_order=model_order)
    plot_figure3_temporal_curves(metrics_long, figures_dir, model_order=model_order)

    geo_topk_by_seed = plot_geo_chicago_topk_frequency(
        predictions=predictions,
        chicago_geo_points=chicago_geo_points,
        out_dir=figures_dir,
        model_order=model_order,
        k=5,
    )
    geo_topk_by_seed.to_csv(metrics_dir / "metrics_geo_chicago_topk_selection_frequency_by_seed.csv", index=False)
    geo_best_model, geo_neighbor_rates = plot_geo_chicago_neighbor_exact(
        predictions=predictions,
        metrics_long=metrics_long,
        chicago_geo_points=chicago_geo_points,
        out_dir=figures_dir,
        model_order=model_order,
    )
    geo_neighbor_rates.to_csv(metrics_dir / "metrics_geo_chicago_neighbor_exact_rates.csv", index=False)
    geo_point_best_model, geo_point_summary = plot_geo_chicago_event_hitmaps(
        predictions=predictions,
        metrics_long=metrics_long,
        chicago_geo_points=chicago_geo_points,
        chicago_geo_events=chicago_geo_events,
        time_window_days=int(time_window_days),
        anchor_date=str(anchor_date),
        out_dir=figures_dir,
        model_order=model_order,
        topk_k=5,
        sample_n=50000,
    )
    geo_point_summary.to_csv(metrics_dir / "metrics_geo_chicago_event_hitmap_summary.csv", index=False)

    topk_by_seed, k_map, region_n_map = compute_topk_coverage(predictions)
    topk_by_seed.to_csv(metrics_dir / "metrics_topk_coverage_by_seed.csv", index=False)
    plot_figure4_topk_coverage(topk_by_seed, figures_dir, model_order=model_order)

    region_bucket_metrics, bucket_order, bucket_region_n = compute_region_bucket_metrics_by_seed(predictions, n_buckets=4)
    region_bucket_metrics.to_csv(metrics_dir / "metrics_region_bucket_by_seed.csv", index=False)
    plot_figure6_heatmap(
        region_bucket_metrics=region_bucket_metrics,
        bucket_order=bucket_order,
        out_dir=figures_dir,
        model_order=model_order,
        metric="auc",
    )
    plot_figure6_region_bucket_heatmap(
        region_bucket_metrics=region_bucket_metrics,
        bucket_order=bucket_order,
        out_dir=figures_dir,
        model_order=model_order,
        metric="f1",
    )
    plot_processed_crime_group_counts(chicago_train_events, chicago_test_events, nibrs_events, figures_dir)

    feature_importance_by_seed.to_csv(metrics_dir / "metrics_feature_importance_by_seed.csv", index=False)
    fi_summary = summarize_feature_importance(feature_importance_by_seed)
    fi_summary.to_csv(metrics_dir / "metrics_feature_importance_summary.csv", index=False)
    fi_fig_name = plot_feature_importance(
        feature_importance_summary=fi_summary,
        out_dir=figures_dir,
        model_order=model_order,
        dataset="chicago_2025",
        top_n=12,
    )

    return {
        "table1_rows": int(len(table1)),
        "table2_rows": int(len(table2_fmt)),
        "k_by_domain": {k: v for k, v in sorted(k_map.items())},
        "region_n_by_domain": {k: int(v) for k, v in sorted(region_n_map.items())},
        "region_bucket_count": int(len(bucket_order)),
        "region_bucket_region_n": int(bucket_region_n),
        "geo_chicago_best_model": geo_best_model,
        "geo_chicago_event_map_model": geo_point_best_model,
        "feature_importance_rows": int(len(fi_summary)),
        "feature_importance_figure": fi_fig_name,
    }


def generate_compare_windows_figure2(
    metrics_long_paths: Dict[int, Path],
    out_dir: Path,
    model_order: Sequence[str],
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    for w, p in metrics_long_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"metrics_long missing for window {w}: {p}")

    all_rows: List[pd.DataFrame] = []
    for w, path in sorted(metrics_long_paths.items()):
        df = pd.read_csv(path)
        df = df[(df["slice"] == "overall") & (df["metric"].isin(["auc", "f1"]))].copy()
        df["time_window_days"] = int(w)
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)

    gap_rows: List[Dict[str, object]] = []
    for (window, model, seed, metric), g in all_df.groupby(["time_window_days", "model", "seed", "metric"]):
        chi = g[g["domain"] == "chicago_2025"]["value"]
        nib = g[g["domain"] == "nibrs"]["value"]
        if chi.empty or nib.empty:
            continue
        gap_rows.append(
            {
                "time_window_days": int(window),
                "model": str(model),
                "seed": int(seed),
                "metric": str(metric),
                "delta": float(chi.iloc[0] - nib.iloc[0]),
            }
        )

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(out_dir / "fig2_gap_values_by_seed.csv", index=False)

    for metric in ["auc", "f1"]:
        sub = gap_df[gap_df["metric"] == metric]
        agg = (
            sub.groupby(["time_window_days", "model"], as_index=False)
            .agg(
                mean=("delta", "mean"),
                std=("delta", "std"),
            )
            .fillna(0.0)
        )

        windows = sorted(agg["time_window_days"].unique().tolist())
        x = np.arange(len(model_order), dtype=float)
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, w in enumerate(windows):
            means = []
            errs = []
            for model in model_order:
                row = agg[(agg["time_window_days"] == w) & (agg["model"] == model)]
                means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
                errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width=width, yerr=errs, capsize=3, label=f"{w}d")

        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20)
        ax.set_ylabel(f"Delta {metric.upper()} (Chicago2025 - NIBRS)")
        ax.set_title(f"Figure 2: Generalization Gap ({metric.upper()}) by Time Window")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Window")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig2_gap_{metric}.png", dpi=180)
        plt.close(fig)

    return {
        "gap_rows": int(len(gap_df)),
        "windows": sorted(metrics_long_paths.keys()),
    }


def generate_compare_windows_overall_metric_bars(
    metrics_long_paths: Dict[int, Path],
    out_dir: Path,
    model_order: Sequence[str],
    metrics: Sequence[str] = ("auc", "f1", "precision", "recall"),
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    for w, p in metrics_long_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"metrics_long missing for window {w}: {p}")

    all_rows: List[pd.DataFrame] = []
    for w, path in sorted(metrics_long_paths.items()):
        df = pd.read_csv(path)
        base = df[(df["slice"] == "overall") & (df["metric"].isin(metrics))].copy()
        base["time_window_days"] = int(w)
        all_rows.append(base)

    if not all_rows:
        return {"metric_rows": 0, "figures": []}

    all_df = pd.concat(all_rows, ignore_index=True)
    agg = (
        all_df.groupby(["time_window_days", "domain", "model", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .fillna(0.0)
    )
    agg.to_csv(out_dir / "fig_compare_windows_overall_values.csv", index=False)

    domain_palette = {
        "chicago_2025": "#E76F51",
        "nibrs": "#457B9D",
    }
    hatch_by_window = {
        7: "",
        3: "///",
    }

    combo_order: List[Tuple[str, int]] = [
        ("chicago_2025", 7),
        ("chicago_2025", 3),
        ("nibrs", 7),
        ("nibrs", 3),
    ]
    combo_order = [c for c in combo_order if c[1] in metrics_long_paths]
    if not combo_order:
        combo_order = [(d, int(w)) for d in DOMAIN_ORDER for w in sorted(metrics_long_paths.keys())]

    x = np.arange(len(model_order), dtype=float)
    width = 0.18
    start = -((len(combo_order) - 1) / 2.0) * width
    figure_files: List[str] = []

    for metric in metrics:
        metric_df = agg[agg["metric"] == metric].copy()
        if metric_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 5.5))
        for idx, (domain, window_days) in enumerate(combo_order):
            means: List[float] = []
            errs: List[float] = []
            for model in model_order:
                row = metric_df[
                    (metric_df["domain"] == domain)
                    & (metric_df["time_window_days"] == int(window_days))
                    & (metric_df["model"] == model)
                ]
                means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
                errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

            offset = start + idx * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=errs,
                capsize=3,
                color=domain_palette.get(domain, "#999999"),
                hatch=hatch_by_window.get(int(window_days), ""),
                edgecolor="black",
                linewidth=0.7,
                alpha=0.95,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Overall {metric.upper()} (3d vs 7d; Chicago2025 vs NIBRS)")
        ax.grid(axis="y", alpha=0.3)

        domain_handles = [
            Patch(facecolor=domain_palette["chicago_2025"], edgecolor="black", label="Chicago2025"),
            Patch(facecolor=domain_palette["nibrs"], edgecolor="black", label="NIBRS"),
        ]
        window_handles = []
        for w in sorted(metrics_long_paths.keys(), reverse=True):
            label = f"{int(w)}d"
            window_handles.append(
                Patch(facecolor="#DDDDDD", edgecolor="black", hatch=hatch_by_window.get(int(w), ""), label=label)
            )
        legend_domain = ax.legend(handles=domain_handles, title="Domain", loc="upper left")
        ax.add_artist(legend_domain)
        ax.legend(handles=window_handles, title="Window", loc="upper right")

        fig.tight_layout()
        fname = f"fig_compare_windows_overall_{metric}.png"
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)
        figure_files.append(fname)

    return {
        "metric_rows": int(len(agg)),
        "figures": sorted(figure_files),
        "windows": sorted(metrics_long_paths.keys()),
    }
