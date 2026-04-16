from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

REQUIRED_COLS = [
    "region_id",
    "time_id",
    "y",
    "y_count",
    "model",
    "dataset",
    "y_score",
    "y_pred",
    "seed",
]


@dataclass
class DeploymentData:
    frame: pd.DataFrame
    output_root: Path
    thresholds: Dict[int, int]
    fixed_seed: int = 0


def _is_valid_output_root(path: Path) -> bool:
    return (
        (path / "7d" / "metrics" / "predictions.csv").exists()
        and (path / "3d" / "metrics" / "predictions.csv").exists()
    )


def list_candidate_output_roots(project_root: Path) -> List[Path]:
    candidates: List[Path] = []

    for p in sorted(project_root.glob("output_timeline_*")):
        if p.is_dir() and _is_valid_output_root(p):
            candidates.append(p)

    fallback = project_root / "outputs"
    if fallback.is_dir() and _is_valid_output_root(fallback):
        candidates.append(fallback)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _load_window_predictions(output_root: Path, window_days: int, fixed_seed: int) -> pd.DataFrame:
    csv_path = output_root / f"{window_days}d" / "metrics" / "predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    frame = frame[REQUIRED_COLS].copy()
    frame = frame[frame["seed"].astype(int) == int(fixed_seed)].copy()

    if frame.empty:
        raise ValueError(f"No rows found for seed={fixed_seed} in {csv_path}")

    frame["time_window_days"] = int(window_days)
    frame["time_id"] = pd.to_datetime(frame["time_id"], errors="coerce").dt.strftime("%Y-%m-%d")

    numeric_cols = ["y", "y_count", "y_score", "y_pred", "seed"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["region_id"] = frame["region_id"].astype(str)
    frame["model"] = frame["model"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)

    frame = frame.dropna(subset=["region_id", "time_id", "y_score", "y_pred", "y_count", "y"])
    frame["y"] = frame["y"].astype(int)
    frame["y_pred"] = frame["y_pred"].astype(int)
    frame["seed"] = frame["seed"].astype(int)

    return frame


def _load_thresholds(output_root: Path) -> Dict[int, int]:
    thresholds: Dict[int, int] = {}
    for w in (7, 3):
        p = output_root / f"{w}d" / "logs" / "run_summary.json"
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if "threshold" in payload:
                thresholds[int(w)] = int(payload["threshold"])
        except Exception:
            continue
    return thresholds


def load_deployment_data(
    project_root: Path,
    output_root: Optional[Path] = None,
    fixed_seed: int = 0,
) -> DeploymentData:
    project_root = project_root.resolve()

    if output_root is None:
        candidates = list_candidate_output_roots(project_root)
        if not candidates:
            raise FileNotFoundError(
                "No valid output root found. Expected folders like output_timeline_*/ with both 7d and 3d predictions.csv."
            )
        output_root = candidates[0]
    else:
        output_root = output_root.resolve()

    if not _is_valid_output_root(output_root):
        raise FileNotFoundError(
            f"Invalid output root: {output_root}. Missing 7d/3d metrics/predictions.csv."
        )

    part_7d = _load_window_predictions(output_root, window_days=7, fixed_seed=fixed_seed)
    part_3d = _load_window_predictions(output_root, window_days=3, fixed_seed=fixed_seed)
    frame = pd.concat([part_7d, part_3d], ignore_index=True)

    thresholds = _load_thresholds(output_root)

    return DeploymentData(frame=frame, output_root=output_root, thresholds=thresholds, fixed_seed=int(fixed_seed))


def get_available_models(data: DeploymentData, window_days: int, dataset: str) -> List[str]:
    subset = data.frame[
        (data.frame["time_window_days"] == int(window_days)) & (data.frame["dataset"] == str(dataset))
    ]
    models = sorted(subset["model"].dropna().unique().tolist())
    return models


def get_available_datasets(data: DeploymentData, window_days: int) -> List[str]:
    subset = data.frame[data.frame["time_window_days"] == int(window_days)]
    return sorted(subset["dataset"].dropna().unique().tolist())


def get_available_regions(data: DeploymentData, window_days: int, dataset: str, model: str) -> List[str]:
    subset = data.frame[
        (data.frame["time_window_days"] == int(window_days))
        & (data.frame["dataset"] == str(dataset))
        & (data.frame["model"] == str(model))
    ]
    return sorted(subset["region_id"].dropna().unique().tolist())


def get_available_times(data: DeploymentData, window_days: int, dataset: str, model: str) -> List[str]:
    subset = data.frame[
        (data.frame["time_window_days"] == int(window_days))
        & (data.frame["dataset"] == str(dataset))
        & (data.frame["model"] == str(model))
    ]
    times = sorted(subset["time_id"].dropna().unique().tolist())
    return times


def get_single_prediction(
    data: DeploymentData,
    window_days: int,
    dataset: str,
    model: str,
    region_id: str,
    time_id: str,
) -> pd.Series:
    subset = data.frame[
        (data.frame["time_window_days"] == int(window_days))
        & (data.frame["dataset"] == str(dataset))
        & (data.frame["model"] == str(model))
        & (data.frame["region_id"] == str(region_id))
        & (data.frame["time_id"] == str(time_id))
    ]

    if subset.empty:
        raise ValueError(
            "No matching prediction found for the selected inputs. "
            "Please check window/dataset/model/region/time combination."
        )

    return subset.iloc[0]


def topk_for_time(
    data: DeploymentData,
    window_days: int,
    dataset: str,
    model: str,
    time_id: str,
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    subset = data.frame[
        (data.frame["time_window_days"] == int(window_days))
        & (data.frame["dataset"] == str(dataset))
        & (data.frame["model"] == str(model))
        & (data.frame["time_id"] == str(time_id))
    ].copy()

    if subset.empty:
        raise ValueError("No rows found for Top-K calculation at selected time.")

    subset = subset.sort_values(["y_score", "region_id"], ascending=[False, True]).reset_index(drop=True)
    subset["rank"] = subset.index + 1

    k_eff = max(1, min(int(k), len(subset)))
    topk = subset.head(k_eff).copy()

    total_incidents = float(subset["y_count"].sum())
    topk_incidents = float(topk["y_count"].sum())
    coverage = (topk_incidents / total_incidents) if total_incidents > 0 else 0.0

    topk_hit_rate = float((topk["y"] == 1).mean()) if len(topk) > 0 else 0.0

    stats = {
        "k": float(k_eff),
        "n_regions": float(len(subset)),
        "coverage_at_k": float(coverage),
        "topk_hit_rate": float(topk_hit_rate),
        "total_incidents": float(total_incidents),
        "topk_incidents": float(topk_incidents),
    }

    keep_cols = ["rank", "region_id", "y_score", "y_pred", "y_count", "y"]
    return topk[keep_cols], stats
