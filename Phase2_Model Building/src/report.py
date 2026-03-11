from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "(no rows)"

    if max_rows is not None:
        df = df.head(max_rows)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows: List[str] = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def explain_domain_shift(delta_auc: float, region_strategy: str, dropped_ratio_nibrs: float) -> str:
    statements = []
    if pd.notna(delta_auc) and delta_auc > 0:
        statements.append(
            f"AUC drops by {delta_auc:.4f} from Chicago to NIBRS, showing measurable domain shift."
        )
    elif pd.notna(delta_auc) and delta_auc < 0:
        statements.append(
            f"Delta AUC is negative ({delta_auc:.4f}), so NIBRS AUC appears higher than Chicago. This can happen with severe class imbalance and smoother spatial aggregation, not necessarily better cross-domain calibration."
        )
    else:
        statements.append("No clear AUC drop is observed for the selected model, but distribution shift still exists.")

    if region_strategy == "county":
        statements.append(
            "Spatial granularity differs from city-level Chicago patterns because the aligned unit is county; this reduces spatial specificity."
        )

    statements.append(
        f"Crime taxonomy harmonization discards part of the raw offenses (NIBRS drop ratio {dropped_ratio_nibrs:.2%}), which removes signal and can lower transfer performance."
    )

    statements.append(
        "NIBRS reporting practices and jurisdiction composition differ from Chicago historical records, causing feature distribution and base-rate mismatch."
    )
    return " ".join(statements)


def generate_phase2_report(
    report_path: Path,
    metadata: Dict[str, object],
    metrics_df: pd.DataFrame,
    delta_auc_df: pd.DataFrame,
    temporal_month_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    region_strategy = str(metadata.get("region_strategy", "unknown"))
    time_window_days = int(metadata.get("time_window_days", 7))
    artifact_root = str(metadata.get("artifact_root", "outputs"))
    threshold = metadata.get("threshold")
    if isinstance(threshold, dict):
        threshold_str = json.dumps(threshold, sort_keys=True)
        threshold_label = "crime-group specific"
    else:
        threshold_str = str(threshold)
        threshold_label = "global"
    chicago_drop = float(metadata.get("chicago_unmapped_ratio", 0.0))
    nibrs_drop = float(metadata.get("nibrs_unmapped_ratio", 0.0))

    best_delta = float("nan")
    if not delta_auc_df.empty:
        best_delta = float(delta_auc_df["delta_auc"].mean())

    explanation = explain_domain_shift(best_delta, region_strategy, nibrs_drop)

    lines: List[str] = []
    lines.append("# IT5006 Phase 2 Report (Auto-Generated)")
    lines.append("")
    lines.append("## Project Background")
    lines.append("- Course project: Predictive Policing / Crime Risk Prediction.")
    lines.append("- Task: Predicting places of increased crime risk with region-level modeling.")
    lines.append("- Training data: Chicago 2015-2024. In-domain test: Chicago 2025. OOD test: NIBRS.")
    lines.append("")
    lines.append("## Data Alignment Decisions")
    lines.append(f"- Region strategy: `{region_strategy}`")
    lines.append(f"- Label threshold type (from Chicago train only): `{threshold_label}`")
    lines.append(f"- Label threshold value: `{threshold_str}`")
    if time_window_days == 7:
        lines.append("- Time unit: weekly (`time_id` = week start date, ISO).")
    else:
        lines.append(f"- Time unit: fixed `{time_window_days}-day` window (`time_id` = anchor-binned ISO date).")
    lines.append("- Label: `y = 1{total_count_{t+1} >= threshold}`")
    lines.append("- Crime groups: theft_larceny, assault_battery, burglary, robbery, motor_vehicle_theft, drug_narcotics.")
    lines.append(f"- Chicago unmapped drop ratio: `{chicago_drop:.2%}`")
    lines.append(f"- NIBRS unmapped drop ratio: `{nibrs_drop:.2%}`")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append(_markdown_table(metrics_df.sort_values(["dataset", "model"])))
    lines.append("")

    lines.append("## Performance Drop (Chicago 2025 vs NIBRS)")
    lines.append(_markdown_table(delta_auc_df))
    lines.append("")

    lines.append("## Temporal Stability (Monthly)")
    lines.append(_markdown_table(temporal_month_df.sort_values(["dataset", "model", "period"]).head(40)))
    lines.append("")

    lines.append("## Spatial Top-K")
    if topk_df.empty:
        lines.append("Spatial Top-K was skipped for datasets with a single region after alignment.")
    else:
        lines.append(_markdown_table(topk_df.sort_values(["dataset", "model", "k"])))
    lines.append("")

    lines.append("## Domain Shift Explanation")
    lines.append(explanation)
    lines.append("")

    lines.append("## Artifact Paths")
    lines.append(f"- Processed tables: `{artifact_root}/data_processed/`")
    lines.append(f"- Models: `{artifact_root}/models/`")
    lines.append(f"- Metrics: `{artifact_root}/metrics/`")
    lines.append(f"- Tables directory: `{artifact_root}/tables/`")
    lines.append(f"- Table 1: `{artifact_root}/tables/table1_distribution.csv`")
    lines.append(f"- Table 2: `{artifact_root}/tables/table2_results.csv`")
    lines.append(f"- Figures directory: `{artifact_root}/figures/`")
    lines.append(f"- Figure 1 AUC: `{artifact_root}/figures/fig1_overall_auc.png`")
    lines.append(f"- Figure 1 F1: `{artifact_root}/figures/fig1_overall_f1.png`")
    lines.append(f"- Figure 1 Precision: `{artifact_root}/figures/fig1_overall_precision.png`")
    lines.append(f"- Figure 1 Recall: `{artifact_root}/figures/fig1_overall_recall.png`")
    lines.append(f"- Figure 3a: `{artifact_root}/figures/fig3a_temporal_chicago2025_f1.png`")
    lines.append(f"- Figure 3b: `{artifact_root}/figures/fig3b_temporal_nibrs_f1.png`")
    lines.append(f"- Figure 4 (Chicago2025): `{artifact_root}/figures/fig4_topk_coverage_chicago2025.png`")
    lines.append(f"- Figure 4 (NIBRS): `{artifact_root}/figures/fig4_topk_coverage_nibrs.png`")
    lines.append(f"- Geo Top-K Map (Chicago): `{artifact_root}/figures/fig_geo_chicago_topk_selection_frequency.png`")
    lines.append(f"- Geo Neighbor-vs-Exact Map (Chicago): `{artifact_root}/figures/fig_geo_chicago_neighbor_vs_exact.png`")
    lines.append(f"- Geo Event Hitmap Exact (Chicago): `{artifact_root}/figures/fig_geo_chicago_event_hitmap_exact.png`")
    lines.append(f"- Geo Event Hitmap Neighbor (Chicago): `{artifact_root}/figures/fig_geo_chicago_event_hitmap_neighbor.png`")
    lines.append(f"- Neighbor-aware F1 (Chicago2025): `{artifact_root}/figures/fig_neighboraware_f1_chicago2025.png`")
    lines.append(f"- Neighbor-aware F1 (NIBRS): `{artifact_root}/figures/fig_neighboraware_f1_nibrs.png`")
    lines.append(f"- Figure 6: `{artifact_root}/figures/fig6_heatmap_nibrs_auc.png`")
    lines.append(f"- Figure 6B: `{artifact_root}/figures/fig6b_heatmap_nibrs_region_bucket_f1.png`")
    lines.append(f"- Feature Importance (by-seed): `{artifact_root}/metrics/metrics_feature_importance_by_seed.csv`")
    lines.append(f"- Feature Importance (summary): `{artifact_root}/metrics/metrics_feature_importance_summary.csv`")
    lines.append(f"- Feature Importance Figure: `{artifact_root}/figures/fig_feature_importance_chicago_2025.png`")
    lines.append(f"- Processed-counts Chicago: `{artifact_root}/figures/fig_processed_counts_chicago.png`")
    lines.append(f"- Processed-counts NIBRS: `{artifact_root}/figures/fig_processed_counts_nibrs.png`")
    lines.append(f"- Logs: `{artifact_root}/logs/pipeline.log`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def compute_generalization_gap(metrics_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"dataset", "model", "auc_roc", "f1"}
    if not required_cols.issubset(metrics_df.columns):
        return pd.DataFrame(
            columns=[
                "model",
                "auc_chicago_2025",
                "auc_nibrs",
                "delta_auc",
                "f1_chicago_2025",
                "f1_nibrs",
                "delta_f1",
            ]
        )

    chi = metrics_df[metrics_df["dataset"] == "chicago_2025"][["model", "auc_roc", "f1"]].rename(
        columns={"auc_roc": "auc_chicago_2025", "f1": "f1_chicago_2025"}
    )
    nibrs = metrics_df[metrics_df["dataset"] == "nibrs"][["model", "auc_roc", "f1"]].rename(
        columns={"auc_roc": "auc_nibrs", "f1": "f1_nibrs"}
    )
    gap = chi.merge(nibrs, on="model", how="inner")
    gap["delta_auc"] = gap["auc_chicago_2025"] - gap["auc_nibrs"]
    gap["delta_f1"] = gap["f1_chicago_2025"] - gap["f1_nibrs"]
    return gap.sort_values("model").reset_index(drop=True)


def generate_phase2_comparison_report(
    report_path: Path,
    window_metrics: Dict[int, pd.DataFrame],
    window_thresholds: Dict[int, object],
    metadata: Dict[str, object],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    region_strategy = str(metadata.get("region_strategy", "unknown"))
    anchor_date = str(metadata.get("anchor_date", "2015-01-01"))
    figure2_dir = str(metadata.get("figure2_dir", "outputs/compare_windows/figures"))
    output_root = str(metadata.get("output_root", "outputs"))

    metrics_rows: List[pd.DataFrame] = []
    gap_rows: List[pd.DataFrame] = []

    for w in sorted(window_metrics.keys()):
        metric = window_metrics[w].copy()
        metric["time_window"] = f"{w}d"
        metrics_rows.append(metric)

        gap = compute_generalization_gap(metric)
        gap["time_window"] = f"{w}d"
        gap_rows.append(gap)

    combined_metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    combined_gaps = pd.concat(gap_rows, ignore_index=True) if gap_rows else pd.DataFrame()

    lines: List[str] = []
    lines.append("# IT5006 Phase 2 Report (Time Window Comparison)")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append("- Two time windows were evaluated with the same preprocessing/mapping/model family:")
    lines.append("  - `7d`: week-start aggregation (original setting).")
    lines.append("  - `3d`: anchor-based fixed bins with shared anchor.")
    lines.append(f"- Shared anchor date for 3-day bins: `{anchor_date}`.")
    lines.append(f"- Region strategy: `{region_strategy}`.")
    lines.append("")

    thr_rows: List[Dict[str, str]] = []
    for w, t in sorted(window_thresholds.items()):
        if isinstance(t, dict):
            thr_rows.append(
                {
                    "time_window": f"{w}d",
                    "threshold_type": "by_crime_group",
                    "threshold": json.dumps(t, sort_keys=True),
                }
            )
        else:
            thr_rows.append(
                {
                    "time_window": f"{w}d",
                    "threshold_type": "global",
                    "threshold": str(t),
                }
            )
    thr_df = pd.DataFrame(thr_rows)
    lines.append("## Thresholds By Time Window")
    lines.append(_markdown_table(thr_df))
    lines.append("")

    lines.append("## Metrics Comparison (Chicago2025 and NIBRS)")
    if combined_metrics.empty:
        lines.append("(no metrics)")
    else:
        show_cols = [
            "time_window",
            "dataset",
            "model",
            "precision",
            "recall",
            "f1",
            "auc_roc",
        ]
        lines.append(_markdown_table(combined_metrics[show_cols].sort_values(["time_window", "dataset", "model"])))
    lines.append("")

    lines.append("## Generalization Gap Comparison (Delta = Chicago2025 - NIBRS)")
    if combined_gaps.empty:
        lines.append("(no gap table)")
    else:
        show_cols = [
            "time_window",
            "model",
            "auc_chicago_2025",
            "auc_nibrs",
            "delta_auc",
            "f1_chicago_2025",
            "f1_nibrs",
            "delta_f1",
        ]
        lines.append(_markdown_table(combined_gaps[show_cols].sort_values(["model", "time_window"])))
    lines.append("")

    lines.append("## Output Directories")
    lines.append(
        f"- `{output_root}/7d/data_processed`, `{output_root}/7d/models`, `{output_root}/7d/metrics`, `{output_root}/7d/figures`"
    )
    lines.append(
        f"- `{output_root}/3d/data_processed`, `{output_root}/3d/models`, `{output_root}/3d/metrics`, `{output_root}/3d/figures`"
    )
    lines.append(f"- Figure 2 (Delta AUC): `{figure2_dir}/fig2_gap_auc.png`")
    lines.append(f"- Figure 2 (Delta F1): `{figure2_dir}/fig2_gap_f1.png`")
    lines.append(f"- Compare Overall AUC: `{figure2_dir}/fig_compare_windows_overall_auc.png`")
    lines.append(f"- Compare Overall F1: `{figure2_dir}/fig_compare_windows_overall_f1.png`")
    lines.append(f"- Compare Overall Precision: `{figure2_dir}/fig_compare_windows_overall_precision.png`")
    lines.append(f"- Compare Overall Recall: `{figure2_dir}/fig_compare_windows_overall_recall.png`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone report generator.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--topk", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    delta = pd.read_csv(args.delta)
    temporal = pd.read_csv(args.temporal)
    topk = pd.read_csv(args.topk)

    generate_phase2_report(
        report_path=args.report,
        metadata={},
        metrics_df=metrics,
        delta_auc_df=delta,
        temporal_month_df=temporal,
        topk_df=topk,
    )


if __name__ == "__main__":
    main()
