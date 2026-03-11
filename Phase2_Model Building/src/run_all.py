from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_io import (
    discover_input_files,
    inspect_csv_schemas,
    list_project_csvs,
    read_csv_flexible,
    resolve_column,
    scan_file_tree,
    setup_logging,
)
from .eval import (
    compute_delta_auc,
    compute_topk_hit_rate,
    plot_temporal_curves,
    plot_topk,
    summarize_classification_metrics,
    temporal_metrics,
)
from .features import (
    add_binary_label,
    aggregate_weekly_events,
    apply_region_train_stats,
    build_supervised_weekly,
    check_schema_consistency,
    compute_threshold,
    enforce_xy_schema,
    fit_region_train_stats,
    make_time_id,
    split_chicago_train_test,
)
from .mapping import (
    build_chicago_mapping_table,
    build_nibrs_mapping_table,
    determine_nibrs_region_strategy,
    map_chicago_primary_type,
    map_nibrs_offense,
)
from .paper_outputs import (
    MODEL_ORDER,
    build_metrics_long,
    compute_overall_metrics_by_seed,
    generate_compare_windows_overall_metric_bars,
    generate_compare_windows_figure2,
    generate_window_tables_and_figures,
)
from .preprocess_chicago import load_chicago_events
from .preprocess_nibrs import load_nibrs_events
from .report import generate_phase2_comparison_report, generate_phase2_report
from .train import (
    compute_feature_importance,
    fit_models,
    normalize_model_selection,
    persist_models,
    predict_with_all_models,
)


def _clean_region_token(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("UNKNOWN").astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    return cleaned.str.upper()


def _load_chicago_geo_points(
    chicago_train_path: Path,
    chicago_test_path: Path,
    state_abbr: str,
    logger,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in [chicago_train_path, chicago_test_path]:
        header = read_csv_flexible(path, nrows=0)
        district_col = resolve_column(header.columns, ["district"])
        lat_col = resolve_column(header.columns, ["latitude", "lat"])
        lon_col = resolve_column(header.columns, ["longitude", "lon", "lng"])
        if district_col is None or lat_col is None or lon_col is None:
            logger.warning("Skipping geo load for %s due to missing district/lat/lon columns.", path.name)
            continue
        geo = read_csv_flexible(
            path,
            usecols=[district_col, lat_col, lon_col],
            low_memory=False,
        ).rename(
            columns={
                district_col: "district",
                lat_col: "latitude",
                lon_col: "longitude",
            }
        )
        frames.append(geo)

    if not frames:
        return pd.DataFrame(columns=["region_id", "latitude", "longitude"])

    geo_all = pd.concat(frames, ignore_index=True)
    geo_all["region_id"] = f"{state_abbr.upper()}__CPD_DIST_" + _clean_region_token(geo_all["district"])
    geo_all["latitude"] = pd.to_numeric(geo_all["latitude"], errors="coerce")
    geo_all["longitude"] = pd.to_numeric(geo_all["longitude"], errors="coerce")
    geo_all = geo_all.dropna(subset=["region_id", "latitude", "longitude"]).copy()
    geo_all = geo_all[~geo_all["region_id"].str.endswith("UNKNOWN")].copy()
    geo_all = geo_all[["region_id", "latitude", "longitude"]]

    logger.info(
        "Chicago geo points loaded rows=%d unique_regions=%d",
        len(geo_all),
        geo_all["region_id"].nunique() if not geo_all.empty else 0,
    )
    return geo_all


def _load_chicago_geo_events(
    chicago_test_path: Path,
    state_abbr: str,
    logger,
) -> pd.DataFrame:
    header = read_csv_flexible(chicago_test_path, nrows=0)
    date_col = resolve_column(header.columns, ["date", "incident_date"])
    district_col = resolve_column(header.columns, ["district"])
    lat_col = resolve_column(header.columns, ["latitude", "lat"])
    lon_col = resolve_column(header.columns, ["longitude", "lon", "lng"])
    if date_col is None or district_col is None or lat_col is None or lon_col is None:
        logger.warning("Chicago geo events unavailable due to missing date/district/lat/lon columns.")
        return pd.DataFrame(columns=["event_date", "region_id", "latitude", "longitude"])

    geo = read_csv_flexible(
        chicago_test_path,
        usecols=[date_col, district_col, lat_col, lon_col],
        low_memory=False,
    ).rename(
        columns={
            date_col: "event_date",
            district_col: "district",
            lat_col: "latitude",
            lon_col: "longitude",
        }
    )
    geo["event_date"] = pd.to_datetime(geo["event_date"], errors="coerce")
    geo["region_id"] = f"{state_abbr.upper()}__CPD_DIST_" + _clean_region_token(geo["district"])
    geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
    geo = geo.dropna(subset=["event_date", "region_id", "latitude", "longitude"]).copy()
    geo = geo[~geo["region_id"].str.endswith("UNKNOWN")].copy()
    geo = geo[["event_date", "region_id", "latitude", "longitude"]]
    logger.info(
        "Chicago geo events loaded rows=%d unique_regions=%d",
        len(geo),
        geo["region_id"].nunique() if not geo.empty else 0,
    )
    return geo


def _compute_chicago_unmapped_ratio(chicago_train: Path, chicago_test: Path) -> float:
    candidates = ["Primary Type", "primary_type", "crime_type"]
    train_col = resolve_column(read_csv_flexible(chicago_train, nrows=0).columns, candidates)
    test_col = resolve_column(read_csv_flexible(chicago_test, nrows=0).columns, candidates)
    if train_col is None or test_col is None:
        raise ValueError("Could not resolve Chicago crime type column when computing unmapped ratio.")

    train = read_csv_flexible(chicago_train, usecols=[train_col], low_memory=False).rename(columns={train_col: "raw_type"})
    test = read_csv_flexible(chicago_test, usecols=[test_col], low_memory=False).rename(columns={test_col: "raw_type"})
    full = pd.concat([train, test], ignore_index=True)
    mapped = full["raw_type"].map(map_chicago_primary_type)
    return float(mapped.isna().mean())


def _compute_nibrs_unmapped_ratio(nibrs_roots: List[Path]) -> float:
    total_rows = 0
    total_unmapped = 0
    for root in nibrs_roots:
        offense = read_csv_flexible(root / "NIBRS_OFFENSE.csv", usecols=["offense_code"], low_memory=False)
        offense_type = read_csv_flexible(
            root / "NIBRS_OFFENSE_TYPE.csv",
            usecols=["offense_code", "offense_name", "offense_category_name"],
            low_memory=False,
        ).drop_duplicates(subset=["offense_code"])

        offense["offense_code"] = offense["offense_code"].astype(str)
        offense_type["offense_code"] = offense_type["offense_code"].astype(str)

        merged = offense.merge(offense_type, on="offense_code", how="left")
        mapped = merged.apply(
            lambda row: map_nibrs_offense(row.get("offense_name"), row.get("offense_category_name"), row.get("offense_code")),
            axis=1,
        )
        total_rows += len(merged)
        total_unmapped += int(mapped.isna().sum())

    return float(total_unmapped / total_rows) if total_rows else 0.0


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)


def _validate_prediction_coverage(
    predictions: pd.DataFrame,
    expected_rows: Dict[str, int],
    selected_models: Sequence[str],
) -> None:
    coverage = (
        predictions.groupby(["seed", "dataset", "model"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    issues: List[str] = []
    selected = set(selected_models)

    for _, row in coverage.iterrows():
        model = str(row["model"])
        if model not in selected:
            continue
        dataset = str(row["dataset"])
        expected = int(expected_rows.get(dataset, -1))
        got = int(row["n_rows"])
        if expected >= 0 and got != expected:
            issues.append(
                f"seed={int(row['seed'])} dataset={dataset} model={model} expected_rows={expected} got_rows={got}"
            )

    if issues:
        preview = "\n".join(issues[:10])
        raise ValueError(
            "Prediction coverage validation failed. Every model must produce predictions on the same number of rows "
            "for each dataset and seed.\n"
            f"{preview}"
        )


def _run_single_window(
    cfg: PipelineConfig,
    window_days: int,
    selected_models: Sequence[str],
    region_strategy,
    chicago_mapping: pd.DataFrame,
    nibrs_mapping: pd.DataFrame,
    chicago_train_events: pd.DataFrame,
    chicago_test_events: pd.DataFrame,
    nibrs_events: pd.DataFrame,
    chicago_geo_points: pd.DataFrame,
    chicago_geo_events: pd.DataFrame,
    chicago_unmapped_ratio: float,
    nibrs_unmapped_ratio: float,
    logger,
) -> Dict[str, object]:
    wtag = f"{int(window_days)}d"
    paths = cfg.ensure_window_dirs(window_days)

    for key in ["data_processed", "models", "metrics", "tables", "figures", "logs"]:
        _clear_directory(paths[key])

    # Save mapping per time-window output for reproducibility.
    chicago_mapping.to_csv(paths["data_processed"] / "crime_group_mapping_chicago.csv", index=False)
    nibrs_mapping.to_csv(paths["data_processed"] / "crime_group_mapping_nibrs.csv", index=False)

    logger.info("[%s] Building panel data", wtag)
    chicago_all_events = pd.concat([chicago_train_events, chicago_test_events], ignore_index=True)
    chicago_panel = aggregate_weekly_events(
        chicago_all_events,
        time_window_days=window_days,
        anchor_date=cfg.anchor_date,
    )
    chicago_supervised_all = build_supervised_weekly(
        chicago_panel,
        time_window_days=window_days,
    )

    chicago_test_time = make_time_id(
        chicago_test_events["event_date"],
        time_window_days=window_days,
        anchor_date=cfg.anchor_date,
    )
    chicago_test_start = pd.Timestamp(chicago_test_time.min())
    chicago_test_end = pd.Timestamp(chicago_test_time.max())

    chicago_train_sup, chicago_2025_sup = split_chicago_train_test(
        supervised_all=chicago_supervised_all,
        chicago_test_start=chicago_test_start,
        chicago_test_end=chicago_test_end,
    )

    nibrs_panel = aggregate_weekly_events(
        nibrs_events,
        time_window_days=window_days,
        anchor_date=cfg.anchor_date,
    )
    nibrs_sup = build_supervised_weekly(
        nibrs_panel,
        time_window_days=window_days,
    )

    logger.info(
        "[%s] Supervised rows | chicago_train=%d chicago_2025=%d nibrs=%d",
        wtag,
        len(chicago_train_sup),
        len(chicago_2025_sup),
        len(nibrs_sup),
    )

    threshold = compute_threshold(
        chicago_train_sup,
        quantile=cfg.threshold_quantile,
    )
    logger.info(
        "[%s] global threshold (Chicago-train only)=%d",
        wtag,
        int(threshold),
    )

    chicago_train_lab = add_binary_label(chicago_train_sup, threshold)
    chicago_2025_lab = add_binary_label(chicago_2025_sup, threshold)
    nibrs_lab = add_binary_label(nibrs_sup, threshold)

    region_stats = fit_region_train_stats(chicago_train_lab)
    chicago_train_lab = apply_region_train_stats(chicago_train_lab, region_stats)
    chicago_2025_lab = apply_region_train_stats(chicago_2025_lab, region_stats)
    nibrs_lab = apply_region_train_stats(nibrs_lab, region_stats)

    chicago_train_xy = enforce_xy_schema(chicago_train_lab, cfg.feature_numeric, "Xy_chicago_train")
    chicago_2025_xy = enforce_xy_schema(chicago_2025_lab, cfg.feature_numeric, "Xy_chicago_2025")
    nibrs_xy = enforce_xy_schema(nibrs_lab, cfg.feature_numeric, "Xy_nibrs")

    schema_check = check_schema_consistency(
        {
            "Xy_chicago_train": chicago_train_xy,
            "Xy_chicago_2025": chicago_2025_xy,
            "Xy_nibrs": nibrs_xy,
        }
    )
    (paths["logs"] / "schema_check.json").write_text(json.dumps(schema_check, indent=2, default=str), encoding="utf-8")

    for name, frame in [
        ("Xy_chicago_train", chicago_train_xy),
        ("Xy_chicago_2025", chicago_2025_xy),
        ("Xy_nibrs", nibrs_xy),
    ]:
        frame.to_parquet(paths["data_processed"] / f"{name}.parquet", index=False)
        frame.to_csv(paths["data_processed"] / f"{name}.csv", index=False)

    seed_values = [int(s) for s in cfg.eval_seeds]
    predictions_by_seed: List[pd.DataFrame] = []
    feature_importance_by_seed_rows: List[pd.DataFrame] = []

    for seed in seed_values:
        logger.info("[%s] Training/evaluating seed=%d", wtag, seed)
        preprocessor, fitted_models, train_logs = fit_models(
            chicago_train_xy=chicago_train_xy,
            numeric_features=cfg.feature_numeric,
            categorical_features=cfg.feature_categorical,
            random_seed=seed,
            selected_models=selected_models,
            lstm_sequence_length=cfg.lstm_sequence_length,
        )
        persist_models(paths["models"] / f"seed_{seed}", preprocessor, fitted_models, train_logs)

        pred_chicago = predict_with_all_models(
            df=chicago_2025_xy,
            preprocessor=preprocessor,
            fitted_models=fitted_models,
            numeric_features=cfg.feature_numeric,
            categorical_features=cfg.feature_categorical,
            persistence_threshold=threshold,
            dataset_name="chicago_2025",
            selected_models=selected_models,
            lstm_sequence_length=cfg.lstm_sequence_length,
        )
        pred_nibrs = predict_with_all_models(
            df=nibrs_xy,
            preprocessor=preprocessor,
            fitted_models=fitted_models,
            numeric_features=cfg.feature_numeric,
            categorical_features=cfg.feature_categorical,
            persistence_threshold=threshold,
            dataset_name="nibrs",
            selected_models=selected_models,
            lstm_sequence_length=cfg.lstm_sequence_length,
        )

        pred = pd.concat([pred_chicago, pred_nibrs], ignore_index=True)
        pred["seed"] = int(seed)
        predictions_by_seed.append(pred)

        fi_seed = compute_feature_importance(
            preprocessor=preprocessor,
            fitted_models=fitted_models,
            reference_df=chicago_2025_xy,
            numeric_features=cfg.feature_numeric,
            categorical_features=cfg.feature_categorical,
            selected_models=selected_models,
            lstm_sequence_length=cfg.lstm_sequence_length,
            seed=int(seed),
            dataset_name="chicago_2025",
            max_lstm_samples=2048,
        )
        if not fi_seed.empty:
            fi_seed["time_window_days"] = int(window_days)
            feature_importance_by_seed_rows.append(fi_seed)

    predictions = pd.concat(predictions_by_seed, ignore_index=True)
    feature_importance_by_seed = (
        pd.concat(feature_importance_by_seed_rows, ignore_index=True)
        if feature_importance_by_seed_rows
        else pd.DataFrame(
            columns=[
                "seed",
                "dataset",
                "model",
                "feature",
                "importance_raw",
                "importance_norm",
                "method",
                "n_reference_rows",
                "n_effective_rows",
                "time_window_days",
            ]
        )
    )
    _validate_prediction_coverage(
        predictions=predictions,
        expected_rows={
            "chicago_2025": int(len(chicago_2025_xy)),
            "nibrs": int(len(nibrs_xy)),
        },
        selected_models=selected_models,
    )
    predictions.to_parquet(paths["metrics"] / "predictions.parquet", index=False)
    predictions.to_csv(paths["metrics"] / "predictions.csv", index=False)

    metrics_seed = compute_overall_metrics_by_seed(predictions)
    metrics_seed.to_csv(paths["metrics"] / "metrics_overall_by_seed.csv", index=False)

    metrics_long = build_metrics_long(predictions, time_window_days=window_days)
    metrics_long.to_csv(paths["metrics"] / "metrics_long.csv", index=False)

    artifact_summary = generate_window_tables_and_figures(
        time_window_days=window_days,
        anchor_date=cfg.anchor_date,
        predictions=predictions,
        metrics_long=metrics_long,
        chicago_train_events=chicago_train_events,
        chicago_test_events=chicago_test_events,
        nibrs_events=nibrs_events,
        chicago_train_xy=chicago_train_xy,
        chicago_2025_xy=chicago_2025_xy,
        nibrs_xy=nibrs_xy,
        chicago_geo_points=chicago_geo_points,
        chicago_geo_events=chicago_geo_events,
        tables_dir=paths["tables"],
        figures_dir=paths["figures"],
        metrics_dir=paths["metrics"],
        feature_importance_by_seed=feature_importance_by_seed,
        selected_models=selected_models,
        seeds=seed_values,
    )

    metrics = (
        metrics_seed.groupby(["domain", "model"], as_index=False)[["precision", "recall", "f1", "auc"]]
        .mean()
        .rename(columns={"domain": "dataset", "auc": "auc_roc"})
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )
    metrics.to_csv(paths["metrics"] / "metrics_overall.csv", index=False)

    temporal_month = (
        metrics_long[(metrics_long["slice"] == "temporal") & (metrics_long["metric"] == "f1")]
        .groupby(["domain", "model", "date_bucket"], as_index=False)["value"]
        .mean()
        .rename(columns={"domain": "dataset", "date_bucket": "period", "value": "f1"})
    )
    temporal_month["n_rows"] = np.nan
    temporal_month["precision"] = np.nan
    temporal_month["recall"] = np.nan
    temporal_month["auc_roc"] = np.nan
    temporal_month = temporal_month[["dataset", "model", "period", "n_rows", "precision", "recall", "f1", "auc_roc"]]
    temporal_month.to_csv(paths["metrics"] / "metrics_temporal_monthly.csv", index=False)

    topk_seed = pd.read_csv(paths["metrics"] / "metrics_topk_coverage_by_seed.csv")
    if topk_seed.empty:
        topk = pd.DataFrame(columns=["dataset", "model", "k", "weekly_count", "topk_hit_rate"])
    else:
        topk = (
            topk_seed.groupby(["domain", "model", "k"], as_index=False)["coverage"]
            .mean()
            .rename(columns={"domain": "dataset", "coverage": "topk_hit_rate"})
        )
        topk["weekly_count"] = np.nan
        topk = topk[["dataset", "model", "k", "weekly_count", "topk_hit_rate"]]
    topk.to_csv(paths["metrics"] / "metrics_topk.csv", index=False)

    delta_auc = compute_delta_auc(metrics)
    delta_auc.to_csv(paths["metrics"] / "metrics_delta_auc.csv", index=False)

    metadata = {
        "region_strategy": region_strategy.strategy,
        "region_strategy_reason": region_strategy.reason,
        "threshold": int(threshold),
        "chicago_unmapped_ratio": chicago_unmapped_ratio,
        "nibrs_unmapped_ratio": nibrs_unmapped_ratio,
        "time_window_days": int(window_days),
        "artifact_root": f"{cfg.output_root_name}/{wtag}",
        "seed_count": len(seed_values),
    }
    generate_phase2_report(
        report_path=paths["root"] / "report_phase2.md",
        metadata=metadata,
        metrics_df=metrics,
        delta_auc_df=delta_auc,
        temporal_month_df=temporal_month,
        topk_df=topk,
    )

    window_summary = {
        "time_window_days": int(window_days),
        "threshold": int(threshold),
        "schema_consistent": bool(schema_check["consistent"]),
        "rows": {
            "Xy_chicago_train": int(len(chicago_train_xy)),
            "Xy_chicago_2025": int(len(chicago_2025_xy)),
            "Xy_nibrs": int(len(nibrs_xy)),
        },
        "selected_models": list(selected_models),
        "seeds": seed_values,
        "anchor_date": cfg.anchor_date,
        "lstm_sequence_length": int(cfg.lstm_sequence_length),
        "artifacts": artifact_summary,
    }
    (paths["logs"] / "run_summary.json").write_text(json.dumps(window_summary, indent=2), encoding="utf-8")

    logger.info("[%s] Window pipeline completed", wtag)

    return {
        "window_days": int(window_days),
        "threshold": int(threshold),
        "metrics": metrics,
        "metrics_long_path": paths["metrics"] / "metrics_long.csv",
        "summary": window_summary,
    }


def run_pipeline(
    project_root: Path,
    selected_models: Optional[List[str]] = None,
    lstm_sequence_length: Optional[int] = None,
    time_windows: Optional[Sequence[int]] = None,
    anchor_date: Optional[str] = None,
    output_root_name: Optional[str] = None,
) -> Dict[str, object]:
    run_start = datetime.now()
    run_start_tag = run_start.strftime("%Y%m%d_%H%M%S")
    run_t0 = time.perf_counter()

    auto_output_root = f"output_timeline_{run_start_tag}"
    cfg = PipelineConfig(
        project_root=project_root,
        output_root_name=(str(output_root_name) if output_root_name else auto_output_root),
    )
    if lstm_sequence_length is not None:
        cfg.lstm_sequence_length = int(lstm_sequence_length)
    if anchor_date is not None:
        cfg.anchor_date = str(anchor_date)

    run_windows = list(time_windows) if time_windows is not None else list(cfg.time_windows)
    run_windows = [int(w) for w in run_windows]
    invalid = [w for w in run_windows if w not in (3, 7)]
    if invalid:
        raise ValueError(f"Invalid time window(s): {invalid}. Supported values are 3 and 7.")

    cfg.ensure_output_dirs()
    selected = normalize_model_selection(selected_models)

    logger = setup_logging(cfg.paths["logs"] / "pipeline.log")
    logger.info("Starting pipeline under %s", project_root)
    logger.info("Run output root: %s", cfg.output_root_name)
    logger.info("Selected models: %s", ", ".join(selected))
    logger.info("Running time windows (days): %s", run_windows)
    logger.info("Anchor date for fixed-window binning: %s", cfg.anchor_date)

    file_tree_df = scan_file_tree(project_root, cfg.paths["logs"] / "file_tree.csv")
    logger.info("Scanned file tree entries=%d", len(file_tree_df))

    input_files = discover_input_files(project_root)
    chicago_train_path: Path = input_files["chicago_train"]  # type: ignore[assignment]
    chicago_test_path: Path = input_files["chicago_test"]  # type: ignore[assignment]
    nibrs_roots: List[Path] = input_files["nibrs_roots"]  # type: ignore[assignment]

    logger.info("Chicago train path: %s", chicago_train_path)
    logger.info("Chicago test path: %s", chicago_test_path)
    logger.info("NIBRS roots: %s", ", ".join(str(p) for p in nibrs_roots))

    all_csvs = list_project_csvs(project_root)
    inspect_csv_schemas(all_csvs, cfg.paths["logs"] / "csv_schema_report.json")
    logger.info("CSV schema snapshot generated for %d files", len(all_csvs))

    region_strategy = determine_nibrs_region_strategy(nibrs_roots)
    logger.info("Region strategy=%s (%s)", region_strategy.strategy, region_strategy.reason)

    chicago_type_col = resolve_column(read_csv_flexible(chicago_train_path, nrows=0).columns, ["Primary Type", "primary_type"])
    if chicago_type_col is None:
        raise ValueError("Could not resolve Chicago primary type column for mapping export")

    chi_train_types = read_csv_flexible(chicago_train_path, usecols=[chicago_type_col], low_memory=False)[chicago_type_col]
    chi_test_types = read_csv_flexible(chicago_test_path, usecols=[chicago_type_col], low_memory=False)[chicago_type_col]
    chicago_mapping = build_chicago_mapping_table(pd.concat([chi_train_types, chi_test_types], ignore_index=True).unique())

    nibrs_offense_type_frames = [
        read_csv_flexible(root / "NIBRS_OFFENSE_TYPE.csv", usecols=["offense_code", "offense_name", "offense_category_name"], low_memory=False)
        for root in nibrs_roots
    ]
    nibrs_mapping = build_nibrs_mapping_table(nibrs_offense_type_frames)

    # Save a root copy for quick inspection.
    chicago_mapping.to_csv(cfg.paths["data_processed"] / "crime_group_mapping_chicago.csv", index=False)
    nibrs_mapping.to_csv(cfg.paths["data_processed"] / "crime_group_mapping_nibrs.csv", index=False)

    chicago_unmapped_ratio = _compute_chicago_unmapped_ratio(chicago_train_path, chicago_test_path)
    nibrs_unmapped_ratio = _compute_nibrs_unmapped_ratio(nibrs_roots)
    logger.info("Chicago unmapped ratio=%.4f", chicago_unmapped_ratio)
    logger.info("NIBRS unmapped ratio=%.4f", nibrs_unmapped_ratio)

    chicago_train_events = load_chicago_events(
        csv_path=chicago_train_path,
        region_strategy=region_strategy,
        state_abbr=cfg.chicago_state_abbr,
        county_fallback=cfg.chicago_county_fallback,
        logger=logger,
    )
    chicago_test_events = load_chicago_events(
        csv_path=chicago_test_path,
        region_strategy=region_strategy,
        state_abbr=cfg.chicago_state_abbr,
        county_fallback=cfg.chicago_county_fallback,
        logger=logger,
    )
    nibrs_events = load_nibrs_events(
        nibrs_roots=nibrs_roots,
        region_strategy=region_strategy,
        logger=logger,
    )
    chicago_geo_points = _load_chicago_geo_points(
        chicago_train_path=chicago_train_path,
        chicago_test_path=chicago_test_path,
        state_abbr=cfg.chicago_state_abbr,
        logger=logger,
    )
    chicago_geo_events = _load_chicago_geo_events(
        chicago_test_path=chicago_test_path,
        state_abbr=cfg.chicago_state_abbr,
        logger=logger,
    )

    logger.info(
        "Canonical event rows | chicago_train=%d chicago_2025=%d nibrs=%d",
        len(chicago_train_events),
        len(chicago_test_events),
        len(nibrs_events),
    )

    window_outputs: List[Dict[str, object]] = []
    metrics_by_window: Dict[int, pd.DataFrame] = {}
    thresholds_by_window: Dict[int, object] = {}
    metrics_long_paths: Dict[int, Path] = {}

    for window_days in run_windows:
        result = _run_single_window(
            cfg=cfg,
            window_days=window_days,
            selected_models=selected,
            region_strategy=region_strategy,
            chicago_mapping=chicago_mapping,
            nibrs_mapping=nibrs_mapping,
            chicago_train_events=chicago_train_events,
            chicago_test_events=chicago_test_events,
            nibrs_events=nibrs_events,
            chicago_geo_points=chicago_geo_points,
            chicago_geo_events=chicago_geo_events,
            chicago_unmapped_ratio=chicago_unmapped_ratio,
            nibrs_unmapped_ratio=nibrs_unmapped_ratio,
            logger=logger,
        )
        window_outputs.append(result)
        metrics_by_window[int(window_days)] = result["metrics"]  # type: ignore[assignment]
        thresholds_by_window[int(window_days)] = result["threshold"]
        metrics_long_paths[int(window_days)] = result["metrics_long_path"]  # type: ignore[assignment]

    for w in run_windows:
        if int(w) not in metrics_long_paths or not metrics_long_paths[int(w)].exists():
            raise FileNotFoundError(f"metrics_long missing for {w}d")

    compare_fig_dir = cfg.outputs_dir / "compare_windows" / "figures"
    _clear_directory(compare_fig_dir)
    compare_info = generate_compare_windows_figure2(
        metrics_long_paths=metrics_long_paths,
        out_dir=compare_fig_dir,
        model_order=[m for m in MODEL_ORDER if m in selected],
    )
    compare_overall_info = generate_compare_windows_overall_metric_bars(
        metrics_long_paths=metrics_long_paths,
        out_dir=compare_fig_dir,
        model_order=[m for m in MODEL_ORDER if m in selected],
    )

    generate_phase2_comparison_report(
        report_path=cfg.outputs_dir / "report_phase2.md",
        window_metrics=metrics_by_window,
        window_thresholds=thresholds_by_window,
        metadata={
            "region_strategy": region_strategy.strategy,
            "anchor_date": cfg.anchor_date,
            "output_root": cfg.output_root_name,
            "figure2_dir": f"{cfg.output_root_name}/compare_windows/figures",
        },
    )

    run_seconds = float(time.perf_counter() - run_t0)
    summary = {
        "run_started_at": run_start.isoformat(timespec="seconds"),
        "run_started_at_tag": run_start_tag,
        "run_duration_seconds": round(run_seconds, 3),
        "output_root": cfg.output_root_name,
        "region_strategy": region_strategy.strategy,
        "region_reason": region_strategy.reason,
        "selected_models": selected,
        "time_windows": run_windows,
        "anchor_date": cfg.anchor_date,
        "lstm_sequence_length": cfg.lstm_sequence_length,
        "chicago_unmapped_ratio": chicago_unmapped_ratio,
        "nibrs_unmapped_ratio": nibrs_unmapped_ratio,
        "windows": [item["summary"] for item in window_outputs],
        "metrics_long_paths": {str(k): str(v) for k, v in metrics_long_paths.items()},
        "compare_windows": compare_info,
        "compare_overall": compare_overall_info,
    }
    (cfg.paths["logs"] / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Pipeline completed successfully for windows=%s", run_windows)
    logger.info("Run duration seconds: %.3f", run_seconds)
    logger.info("Artifacts saved under: %s", cfg.outputs_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Phase 2 predictive policing pipeline.")
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--models", nargs="+", default=["all"], help="all | baseline | logistic | rf | lstm")
    parser.add_argument("--lstm_sequence_length", type=int, default=8)
    parser.add_argument("--anchor_date", default="2015-01-01")
    parser.add_argument(
        "--output_root_name",
        default=None,
        help="Optional output root folder name under project root. Default: output_timeline_<YYYYMMDD_HHMMSS>",
    )
    parser.add_argument(
        "--time_window_days",
        type=int,
        choices=[3, 7],
        default=None,
        help="Run only one window if provided; default runs both 7 and 3 days.",
    )
    args = parser.parse_args()

    run_pipeline(
        args.project_root.resolve(),
        selected_models=args.models,
        lstm_sequence_length=args.lstm_sequence_length,
        time_windows=[args.time_window_days] if args.time_window_days is not None else None,
        anchor_date=args.anchor_date,
        output_root_name=args.output_root_name,
    )


if __name__ == "__main__":
    main()
