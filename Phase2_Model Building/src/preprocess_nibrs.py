from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from .data_io import read_csv_flexible
from .mapping import RegionStrategy, map_nibrs_offense


def _clean_region_token(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("UNKNOWN").astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    return cleaned.str.upper()


def _build_region_id(df: pd.DataFrame, strategy: RegionStrategy) -> pd.Series:
    state = _clean_region_token(df.get("state_abbr", pd.Series("UNK", index=df.index)))

    if strategy.strategy == "county":
        county = _clean_region_token(df.get("county_name", pd.Series("UNKNOWN", index=df.index)))
        region = state + "__" + county

        if strategy.fallback_field and strategy.fallback_field in df.columns:
            fallback = _clean_region_token(df[strategy.fallback_field])
            region = region.where(county != "UNKNOWN", state + "__" + fallback)
        return region

    if strategy.strategy == "agency":
        primary = _clean_region_token(df.get(strategy.primary_field, pd.Series("UNKNOWN", index=df.index)))
        if strategy.fallback_field and strategy.fallback_field in df.columns:
            fallback = _clean_region_token(df[strategy.fallback_field])
            primary = primary.where(primary != "UNKNOWN", fallback)
        return state + "__" + primary

    agency = _clean_region_token(df.get("agency_id", pd.Series("UNKNOWN", index=df.index)))
    return state + "__" + agency


def _read_nibrs_root(root: Path, strategy: RegionStrategy) -> pd.DataFrame:
    incident_cols = ["incident_id", "agency_id", "incident_date"]
    offense_cols = ["incident_id", "offense_code"]
    offense_type_cols = ["offense_code", "offense_name", "offense_category_name"]

    agencies_cols = ["agency_id", "state_abbr", "state_name", "county_name", "ori", "pub_agency_unit", "pub_agency_name"]

    incidents = read_csv_flexible(root / "NIBRS_incident.csv", usecols=incident_cols, low_memory=False)
    offenses = read_csv_flexible(root / "NIBRS_OFFENSE.csv", usecols=offense_cols, low_memory=False)
    offense_type = read_csv_flexible(root / "NIBRS_OFFENSE_TYPE.csv", usecols=offense_type_cols, low_memory=False)
    agencies = read_csv_flexible(root / "agencies.csv", usecols=[c for c in agencies_cols if c], low_memory=False)

    offense_type = offense_type.drop_duplicates(subset=["offense_code"])
    offense_type["offense_code"] = offense_type["offense_code"].astype(str)
    offenses["offense_code"] = offenses["offense_code"].astype(str)

    merged = (
        offenses.merge(incidents, on="incident_id", how="left")
        .merge(offense_type, on="offense_code", how="left")
        .merge(agencies, on="agency_id", how="left")
    )

    merged["event_date"] = pd.to_datetime(merged["incident_date"], errors="coerce")
    merged["crime_group"] = merged.apply(
        lambda row: map_nibrs_offense(row.get("offense_name"), row.get("offense_category_name"), row.get("offense_code")),
        axis=1,
    )
    merged["region_id"] = _build_region_id(merged, strategy)
    merged["raw_offense"] = merged["offense_name"]
    merged["source_state"] = merged.get("state_abbr", "UNK")

    return merged[["event_date", "region_id", "crime_group", "raw_offense", "offense_code", "source_state"]]


def load_nibrs_events(
    nibrs_roots: Sequence[Path],
    region_strategy: RegionStrategy,
    logger: logging.Logger,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for root in nibrs_roots:
        root_frame = _read_nibrs_root(root, region_strategy)
        frames.append(root_frame)
        logger.info("Loaded NIBRS root=%s rows=%d", root, len(root_frame))

    full = pd.concat(frames, ignore_index=True)

    logger.info("NIBRS combined rows=%d", len(full))
    logger.info("NIBRS invalid date rate=%.4f", full["event_date"].isna().mean())
    logger.info("NIBRS unmapped crime rate=%.4f", full["crime_group"].isna().mean())
    logger.info("NIBRS unique regions=%d", full["region_id"].nunique())

    cleaned = full.dropna(subset=["event_date", "crime_group", "region_id"]).copy()
    return cleaned


def summarize_nibrs_schema(root: Path) -> Dict[str, object]:
    incident = read_csv_flexible(root / "NIBRS_incident.csv", nrows=3)
    offense = read_csv_flexible(root / "NIBRS_OFFENSE.csv", nrows=3)
    offense_type = read_csv_flexible(root / "NIBRS_OFFENSE_TYPE.csv", nrows=3)
    agencies = read_csv_flexible(root / "agencies.csv", nrows=3)

    return {
        "root": str(root),
        "incident_columns": incident.columns.tolist(),
        "offense_columns": offense.columns.tolist(),
        "offense_type_columns": offense_type.columns.tolist(),
        "agencies_columns": agencies.columns.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NIBRS roots into canonical event records.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region_strategy", choices=["county", "agency", "state_agency"], default="county")
    parser.add_argument("--primary_field", default="county_name")
    parser.add_argument("--fallback_field", default="ori")
    args = parser.parse_args()

    logger = logging.getLogger("preprocess_nibrs")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    strategy = RegionStrategy(
        strategy=args.region_strategy,
        primary_field=args.primary_field,
        fallback_field=args.fallback_field,
        reason="standalone CLI",
    )

    events = load_nibrs_events(args.roots, strategy, logger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
