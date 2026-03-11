from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import argparse

import pandas as pd

from .data_io import read_csv_flexible


CRIME_GROUPS = [
    "theft_larceny",
    "assault_battery",
    "burglary",
    "robbery",
    "motor_vehicle_theft",
    "drug_narcotics",
]


@dataclass
class RegionStrategy:
    strategy: str
    primary_field: str
    fallback_field: Optional[str]
    reason: str


def map_chicago_primary_type(raw_type: object) -> Optional[str]:
    value = str(raw_type).strip().upper()
    if value in {"THEFT"}:
        return "theft_larceny"
    if value in {"ASSAULT", "BATTERY", "INTIMIDATION"}:
        return "assault_battery"
    if value in {"BURGLARY"}:
        return "burglary"
    if value in {"ROBBERY"}:
        return "robbery"
    if value in {"MOTOR VEHICLE THEFT"}:
        return "motor_vehicle_theft"
    if value in {"NARCOTICS", "OTHER NARCOTIC VIOLATION"}:
        return "drug_narcotics"
    return None


def map_nibrs_offense(
    offense_name: object,
    offense_category_name: object,
    offense_code: object,
) -> Optional[str]:
    category = str(offense_category_name).upper()
    code = str(offense_code).upper()

    if "LARCENY/THEFT" in category:
        return "theft_larceny"
    if "ASSAULT OFFENSES" in category:
        return "assault_battery"
    if "BURGLARY/BREAKING & ENTERING" in category or code == "220":
        return "burglary"
    if category == "ROBBERY" or code == "120":
        return "robbery"
    if "MOTOR VEHICLE THEFT" in category or code == "240":
        return "motor_vehicle_theft"
    if "DRUG/NARCOTIC OFFENSES" in category:
        return "drug_narcotics"

    # Name-level defensive fallback when category is inconsistent.
    name = str(offense_name).upper()
    if "ASSAULT" in name:
        return "assault_battery"
    if "LARCENY" in name or "THEFT" in name:
        return "theft_larceny"
    if "BURGLARY" in name:
        return "burglary"
    if "ROBBERY" in name:
        return "robbery"
    if "MOTOR VEHICLE THEFT" in name:
        return "motor_vehicle_theft"
    if "DRUG" in name or "NARCOTIC" in name:
        return "drug_narcotics"

    return None


def build_chicago_mapping_table(raw_types: Iterable[object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for value in sorted({str(x) for x in raw_types if pd.notna(x)}):
        rows.append(
            {
                "chicago_raw_type": value,
                "crime_group": map_chicago_primary_type(value),
            }
        )
    return pd.DataFrame(rows)


def build_nibrs_mapping_table(offense_type_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(offense_type_frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["offense_code", "offense_name", "offense_category_name"])

    rows: List[Dict[str, object]] = []
    for _, row in merged.sort_values(["offense_code", "offense_name"]).iterrows():
        rows.append(
            {
                "nibrs_raw_offense": row["offense_name"],
                "offense_code": row["offense_code"],
                "offense_category_name": row["offense_category_name"],
                "crime_group": map_nibrs_offense(
                    row["offense_name"],
                    row["offense_category_name"],
                    row["offense_code"],
                ),
            }
        )
    return pd.DataFrame(rows)


def determine_nibrs_region_strategy(nibrs_roots: Sequence[Path]) -> RegionStrategy:
    agency_profiles: List[pd.DataFrame] = []
    for root in nibrs_roots:
        agencies = read_csv_flexible(root / "agencies.csv", low_memory=False)
        agency_profiles.append(agencies)

    combined = pd.concat(agency_profiles, ignore_index=True)
    columns = {c.lower(): c for c in combined.columns}

    # Prefer agency-level alignment so Chicago districts can be matched to jurisdiction-like units.
    for field in ("agency_id", "ori", "pub_agency_unit", "pub_agency_name"):
        if field in combined.columns and combined[field].notna().any():
            fallback = None
            if field != "agency_id" and "agency_id" in combined.columns:
                fallback = "agency_id"
            elif field != "ori" and "ori" in combined.columns:
                fallback = "ori"
            return RegionStrategy(
                strategy="agency",
                primary_field=field,
                fallback_field=fallback,
                reason=f"Selected agency-level alignment ({field}) to match Chicago district-level jurisdictions.",
            )

    has_county = "county_name" in combined.columns and combined["county_name"].notna().any()
    if has_county:
        return RegionStrategy(
            strategy="county",
            primary_field="county_name",
            fallback_field="agency_id" if "agency_id" in combined.columns else None,
            reason="Agency descriptors unavailable; falling back to county-level alignment.",
        )

    return RegionStrategy(
        strategy="state_agency",
        primary_field="state_abbr" if "state_abbr" in columns else "state_name",
        fallback_field="agency_id",
        reason="county and agency descriptors unavailable; using state + agency composite fallback.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export mapping tables and inferred NIBRS region strategy.")
    parser.add_argument("--nibrs_roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = determine_nibrs_region_strategy(args.nibrs_roots)

    nibrs_offense_type_frames = [
        read_csv_flexible(root / "NIBRS_OFFENSE_TYPE.csv", usecols=["offense_code", "offense_name", "offense_category_name"], low_memory=False)
        for root in args.nibrs_roots
    ]
    nibrs_mapping = build_nibrs_mapping_table(nibrs_offense_type_frames)
    nibrs_mapping.to_csv(out_dir / "crime_group_mapping_nibrs.csv", index=False)

    (out_dir / "region_strategy.txt").write_text(
        f"strategy={strategy.strategy}\nprimary_field={strategy.primary_field}\nfallback={strategy.fallback_field}\nreason={strategy.reason}\n",
        encoding="utf-8",
    )
    print(f"Exported mapping and strategy under {out_dir}")


if __name__ == "__main__":
    main()
