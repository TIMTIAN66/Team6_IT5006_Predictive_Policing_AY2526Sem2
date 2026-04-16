from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import CHICAGO_REGION_CANDIDATE_PRIORITY, CRIME_TYPE_COLUMN_CANDIDATES, DATE_COLUMN_CANDIDATES
from .data_io import read_csv_flexible, resolve_column
from .mapping import RegionStrategy, map_chicago_primary_type


def _choose_chicago_region_column(columns: pd.Index) -> Optional[str]:
    for candidate in CHICAGO_REGION_CANDIDATE_PRIORITY:
        match = resolve_column(columns, [candidate])
        if match is not None:
            return match
    return None


def _clean_region_token(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("UNKNOWN").astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    return cleaned.str.upper()


def load_chicago_events(
    csv_path: Path,
    region_strategy: RegionStrategy,
    state_abbr: str,
    county_fallback: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    header = read_csv_flexible(csv_path, nrows=0)

    date_col = resolve_column(header.columns, DATE_COLUMN_CANDIDATES)
    type_col = resolve_column(header.columns, CRIME_TYPE_COLUMN_CANDIDATES)
    if date_col is None or type_col is None:
        raise ValueError(f"Could not resolve date/type columns in {csv_path}")

    region_col = _choose_chicago_region_column(header.columns)

    usecols = [date_col, type_col]
    if region_col is not None:
        usecols.append(region_col)

    raw = read_csv_flexible(csv_path, usecols=usecols, low_memory=False)
    raw = raw.rename(columns={date_col: "event_date", type_col: "raw_type"})

    raw["event_date"] = pd.to_datetime(raw["event_date"], errors="coerce")
    raw["crime_group"] = raw["raw_type"].map(map_chicago_primary_type)

    if region_strategy.strategy == "county":
        raw["region_id"] = f"{state_abbr.upper()}__{county_fallback.upper()}"
        region_mode = "constant county"
    elif region_strategy.strategy == "agency":
        if region_col is None:
            raise ValueError("NIBRS selected agency strategy, but Chicago has no jurisdiction-like column.")
        raw["region_id"] = "IL__CPD_DIST_" + _clean_region_token(raw[region_col])
        region_mode = f"{region_col}"
    else:
        if region_col is None:
            raw["region_id"] = f"{state_abbr.upper()}__{county_fallback.upper()}"
            region_mode = "state+county fallback"
        else:
            raw["region_id"] = raw[region_col].fillna("UNKNOWN").astype(str)
            raw["region_id"] = f"{state_abbr.upper()}__" + raw["region_id"]
            region_mode = f"state+{region_col}"

    total_rows = len(raw)
    invalid_date = raw["event_date"].isna().mean()
    dropped_unmapped = raw["crime_group"].isna().mean()

    logger.info("Chicago file=%s rows=%d", csv_path.name, total_rows)
    logger.info("Chicago date parse null rate=%.4f", invalid_date)
    logger.info("Chicago unmapped crime type rate=%.4f", dropped_unmapped)
    logger.info("Chicago region mode=%s unique_regions=%d", region_mode, raw["region_id"].nunique())

    out = raw.dropna(subset=["event_date", "crime_group", "region_id"]).copy()
    out = out[["event_date", "region_id", "crime_group", "raw_type"]]
    return out


def summarize_chicago_schema(csv_path: Path) -> Dict[str, object]:
    sample = read_csv_flexible(csv_path, nrows=3)
    return {
        "file": str(csv_path),
        "columns": sample.columns.tolist(),
        "preview": sample.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a Chicago crime CSV to canonical event records.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region_strategy", choices=["county", "agency", "state_agency"], default="county")
    parser.add_argument("--state_abbr", default="IL")
    parser.add_argument("--county", default="COOK")
    args = parser.parse_args()

    logger = logging.getLogger("preprocess_chicago")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    strategy = RegionStrategy(
        strategy=args.region_strategy,
        primary_field="county_name",
        fallback_field=None,
        reason="standalone CLI",
    )
    events = load_chicago_events(
        csv_path=args.input,
        region_strategy=strategy,
        state_abbr=args.state_abbr,
        county_fallback=args.county,
        logger=logger,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
