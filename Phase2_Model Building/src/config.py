from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class PipelineConfig:
    project_root: Path
    output_root_name: str = "outputs"
    random_seed: int = 42
    threshold_quantile: float = 0.75
    week_frequency: str = "W-MON"
    topk_values: Tuple[int, ...] = (1, 3, 5)
    lstm_sequence_length: int = 8
    anchor_date: str = "2015-01-01"
    time_windows: Tuple[int, ...] = (7, 3)
    eval_seeds: Tuple[int, ...] = (0, 1, 2)

    chicago_county_fallback: str = "COOK"
    chicago_state_abbr: str = "IL"

    feature_numeric: List[str] = field(
        default_factory=lambda: [
            "lag_0",
            "lag_1",
            "lag_2",
            "lag_4",
            "rolling_mean_4",
            "rolling_sum_4",
            "ewma_4",
            "neighbor_lag_0",
            "neighbor_lag_1",
            "neighbor_roll_mean_4",
            "neighbor_ewma_4",
            "self_neighbor_ratio_lag0",
            "lag1_share_theft_larceny",
            "lag1_share_assault_battery",
            "lag1_share_burglary",
            "lag1_share_robbery",
            "lag1_share_motor_vehicle_theft",
            "lag1_share_drug_narcotics",
            "month",
            "weekofyear",
            "quarter",
            "sin_weekofyear",
            "cos_weekofyear",
            "region_train_mean",
            "region_train_std",
            "region_train_count",
            "region_seen_in_train",
        ]
    )
    feature_categorical: List[str] = field(default_factory=list)

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / self.output_root_name

    @property
    def paths(self) -> Dict[str, Path]:
        out = self.outputs_dir
        return {
            "data_processed": out / "data_processed",
            "models": out / "models",
            "metrics": out / "metrics",
            "figures": out / "figures",
            "logs": out / "logs",
        }

    def ensure_output_dirs(self) -> None:
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def window_root(self, time_window_days: int) -> Path:
        return self.outputs_dir / f"{int(time_window_days)}d"

    def window_paths(self, time_window_days: int) -> Dict[str, Path]:
        root = self.window_root(time_window_days)
        return {
            "root": root,
            "data_processed": root / "data_processed",
            "models": root / "models",
            "metrics": root / "metrics",
            "tables": root / "tables",
            "figures": root / "figures",
            "logs": root / "logs",
        }

    def ensure_window_dirs(self, time_window_days: int) -> Dict[str, Path]:
        p = self.window_paths(time_window_days)
        for key, path in p.items():
            if key == "root":
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
        return p


DATE_COLUMN_CANDIDATES = [
    "date",
    "incident_date",
    "datetime",
    "reported_date",
    "report_date",
]

CRIME_TYPE_COLUMN_CANDIDATES = [
    "primary type",
    "primary_type",
    "crime_type",
    "offense_name",
    "offense_type",
    "offense",
]

CHICAGO_REGION_CANDIDATE_PRIORITY = [
    "district",
    "beat",
    "community area",
    "community_area",
    "ward",
]
