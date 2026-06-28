from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_FILES = {
    "assessments": "assessments.csv",
    "courses": "courses.csv",
    "student_assessment": "studentAssessment.csv",
    "student_info": "studentInfo.csv",
    "student_registration": "studentRegistration.csv",
    "student_vle": "studentVle.csv",
    "vle": "vle.csv",
}


def resolve_csv(data_dir: str | Path, file_name: str) -> Path:
    """Resolve a CSV whether it is stored directly or inside a same-named folder."""
    data_path = Path(data_dir).expanduser().resolve()
    direct = data_path / file_name

    if direct.is_file():
        return direct

    nested = direct / file_name
    if nested.is_file():
        return nested

    matches = sorted(path for path in data_path.rglob(file_name) if path.is_file())
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {file_name!r} in {data_path}. Expected either "
        f"{direct} or {nested}."
    )


def read_csv(data_dir: str | Path, key: str, **kwargs) -> pd.DataFrame:
    """Read one known project CSV by logical key."""
    if key not in REQUIRED_FILES:
        valid = ", ".join(sorted(REQUIRED_FILES))
        raise KeyError(f"Unknown dataset key {key!r}. Valid keys: {valid}")

    return pd.read_csv(resolve_csv(data_dir, REQUIRED_FILES[key]), **kwargs)


def available_csvs(data_dir: str | Path) -> dict[str, Path]:
    """Return resolved paths for all required CSVs that are present."""
    paths: dict[str, Path] = {}
    for key, file_name in REQUIRED_FILES.items():
        try:
            paths[key] = resolve_csv(data_dir, file_name)
        except FileNotFoundError:
            continue
    return paths


def validate_required_files(data_dir: str | Path, required: Iterable[str] | None = None) -> None:
    """Raise a clear error if any required input table is missing."""
    required_keys = list(required or REQUIRED_FILES)
    missing = []
    for key in required_keys:
        try:
            resolve_csv(data_dir, REQUIRED_FILES[key])
        except FileNotFoundError:
            missing.append(REQUIRED_FILES[key])

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required CSV file(s): {missing_text}")
