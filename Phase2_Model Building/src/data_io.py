from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import argparse

import pandas as pd


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("phase2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def normalize_colname(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = {normalize_colname(col): col for col in columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in normalized:
            return normalized[key]
    return None


def read_csv_flexible(
    path: Path,
    usecols: Optional[Sequence[str]] = None,
    nrows: Optional[int] = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(
                path,
                usecols=usecols,
                nrows=nrows,
                low_memory=low_memory,
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=low_memory)


def discover_input_files(project_root: Path) -> Dict[str, object]:
    raw_root = project_root / "data" / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing data/raw under {project_root}")

    chicago_train = None
    chicago_test = None
    for csv_path in sorted(raw_root.rglob("*.csv")):
        name = csv_path.name.lower()
        if "crime_train" in name and "2015" in name:
            chicago_train = csv_path
        elif "crime_test" in name and "2025" in name:
            chicago_test = csv_path

    if chicago_train is None or chicago_test is None:
        raise FileNotFoundError("Could not auto-detect Chicago train/test CSV files.")

    nibrs_roots: List[Path] = []
    for incident_path in sorted(raw_root.rglob("NIBRS_incident.csv")):
        base = incident_path.parent
        required = [
            base / "NIBRS_OFFENSE.csv",
            base / "NIBRS_OFFENSE_TYPE.csv",
            base / "agencies.csv",
        ]
        if all(p.exists() for p in required):
            nibrs_roots.append(base)

    if not nibrs_roots:
        raise FileNotFoundError("Could not find NIBRS roots with required tables.")

    return {
        "chicago_train": chicago_train,
        "chicago_test": chicago_test,
        "nibrs_roots": nibrs_roots,
    }


def scan_file_tree(project_root: Path, output_path: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for path in sorted(project_root.rglob("*")):
        rel = path.relative_to(project_root)
        if path.is_file():
            size = path.stat().st_size
            kind = "file"
        else:
            size = None
            kind = "dir"
        records.append(
            {
                "path": str(rel),
                "kind": kind,
                "size_bytes": size,
            }
        )

    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def inspect_csv_schemas(csv_paths: Sequence[Path], output_json: Path, nrows: int = 3) -> List[Dict[str, object]]:
    report: List[Dict[str, object]] = []
    for csv_path in csv_paths:
        item: Dict[str, object] = {
            "file": str(csv_path),
            "size_bytes": csv_path.stat().st_size,
        }
        try:
            df = read_csv_flexible(csv_path, nrows=nrows, low_memory=False)
            item["columns"] = df.columns.tolist()
            item["preview"] = df.to_dict(orient="records")
            item["ncols"] = len(df.columns)
        except Exception as exc:  # pragma: no cover - defensive logging path
            item["error"] = str(exc)
        report.append(item)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def list_project_csvs(project_root: Path) -> List[Path]:
    return sorted((project_root / "data" / "raw").rglob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan project files and capture CSV schema snapshots.")
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    logs_dir = root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    scan_file_tree(root, logs_dir / "file_tree.csv")
    csvs = list_project_csvs(root)
    inspect_csv_schemas(csvs, logs_dir / "csv_schema_report.json")
    print(f"Scanned {len(csvs)} CSV files and wrote logs to {logs_dir}")


if __name__ == "__main__":
    main()
