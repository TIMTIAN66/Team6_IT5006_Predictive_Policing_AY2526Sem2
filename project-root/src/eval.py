from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def summarize_classification_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (dataset, model), group in predictions.groupby(["dataset", "model"]):
        y_true = group["y"].astype(int)
        y_pred = group["y_pred"].astype(int)
        y_score = group["y_score"].astype(float)

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n_rows": int(len(group)),
                "positive_rate": float(y_true.mean()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "auc_roc": safe_auc(y_true, y_score),
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def temporal_metrics(predictions: pd.DataFrame, period: str) -> pd.DataFrame:
    if period not in {"M", "Q"}:
        raise ValueError("period must be 'M' or 'Q'")

    frame = predictions.copy()
    frame["time_id"] = pd.to_datetime(frame["time_id"])
    frame["period"] = frame["time_id"].dt.to_period(period).astype(str)

    rows: List[Dict[str, object]] = []
    for (dataset, model, bucket), group in frame.groupby(["dataset", "model", "period"]):
        y_true = group["y"].astype(int)
        y_pred = group["y_pred"].astype(int)
        y_score = group["y_score"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "period": bucket,
                "n_rows": int(len(group)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "auc_roc": safe_auc(y_true, y_score),
            }
        )
    return pd.DataFrame(rows)


def plot_temporal_curves(
    temporal_df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    subset = temporal_df[temporal_df["dataset"] == dataset_name].copy()
    if subset.empty:
        return

    periods = sorted(subset["period"].unique().tolist())

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for model, group in subset.groupby("model"):
        model_series = group.set_index("period").reindex(periods)
        axes[0].plot(periods, model_series["f1"], marker="o", label=model)
        axes[1].plot(periods, model_series["auc_roc"], marker="o", label=model)

    axes[0].set_title(f"{dataset_name}: Monthly F1")
    axes[0].set_ylabel("F1")
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"{dataset_name}: Monthly AUC-ROC")
    axes[1].set_ylabel("AUC")
    axes[1].set_xlabel("Period")
    axes[1].grid(alpha=0.3)

    axes[0].legend(loc="best")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_topk_hit_rate(predictions: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (dataset, model), group in predictions.groupby(["dataset", "model"]):
        agg = (
            group.groupby(["time_id", "region_id"], as_index=False)
            .agg(y_score=("y_score", "max"), y_true=("y", "max"))
            .copy()
        )
        n_regions = agg["region_id"].nunique()
        if n_regions <= 1:
            continue

        for k in k_values:
            effective_k = min(k, n_regions)
            weekly_scores: List[float] = []
            for _, weekly in agg.groupby("time_id"):
                top = weekly.nlargest(effective_k, "y_score")
                weekly_scores.append(float(top["y_true"].mean()))

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "k": int(effective_k),
                    "weekly_count": int(len(weekly_scores)),
                    "topk_hit_rate": float(np.mean(weekly_scores)) if weekly_scores else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def plot_topk(topk_df: pd.DataFrame, dataset_name: str, output_path: Path) -> None:
    subset = topk_df[topk_df["dataset"] == dataset_name].copy()
    if subset.empty:
        return

    ks = sorted(subset["k"].unique().tolist())
    models = sorted(subset["model"].unique().tolist())

    x = np.arange(len(models))
    width = 0.8 / max(1, len(ks))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(ks):
        vals = []
        for model in models:
            v = subset[(subset["model"] == model) & (subset["k"] == k)]["topk_hit_rate"]
            vals.append(float(v.iloc[0]) if not v.empty else np.nan)
        ax.bar(x + i * width, vals, width=width, label=f"K={k}")

    ax.set_xticks(x + width * (len(ks) - 1) / 2)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Top-K Hit Rate")
    ax.set_title(f"{dataset_name}: Spatial Top-K Hit Rate")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_delta_auc(metrics_df: pd.DataFrame) -> pd.DataFrame:
    chi = metrics_df[metrics_df["dataset"] == "chicago_2025"][["model", "auc_roc"]].rename(
        columns={"auc_roc": "auc_chicago_2025"}
    )
    nibrs = metrics_df[metrics_df["dataset"] == "nibrs"][["model", "auc_roc"]].rename(
        columns={"auc_roc": "auc_nibrs"}
    )
    merged = chi.merge(nibrs, on="model", how="inner")
    merged["delta_auc"] = merged["auc_chicago_2025"] - merged["auc_nibrs"]
    return merged.sort_values("delta_auc", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone evaluation script for prediction files.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out_metrics", type=Path, required=True)
    args = parser.parse_args()

    if args.predictions.suffix.lower() == ".parquet":
        pred = pd.read_parquet(args.predictions)
    else:
        pred = pd.read_csv(args.predictions)

    metrics = summarize_classification_metrics(pred)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_metrics, index=False)


if __name__ == "__main__":
    main()
