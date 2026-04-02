"""Prepare ecobee raw CSV files for ThermoStill training."""

from pathlib import Path
import shutil

import pandas as pd

from utils import process_house_data


SRC_ROOT = Path("./house_data_csvs0")
DST_ROOT = Path("./house_data_by_state")
STATE_COLUMN = "State"
VALID_STATES = {"CA", "IL", "NY", "TX"}


def prepare_output_directories(dst_root: Path, valid_states: set[str]) -> None:
    """Create one output subdirectory per supported state."""
    for state in valid_states:
        (dst_root / state).mkdir(parents=True, exist_ok=True)


def infer_state(df: pd.DataFrame, csv_path: Path) -> str | None:
    """Return the unique state for a file, or ``None`` if the file should be skipped."""
    if STATE_COLUMN not in df.columns:
        print(f"[SKIP] Missing '{STATE_COLUMN}' column: {csv_path}")
        return None

    states = set(df[STATE_COLUMN].dropna().unique())
    if len(states) != 1:
        print(f"[WARN] Expected exactly one state in {csv_path}, found {states}. Skipping.")
        return None

    state = states.pop()
    if state not in VALID_STATES:
        print(f"[SKIP] Unsupported state '{state}' in {csv_path}")
        return None
    return state


def process_tree(src_root: Path, dst_root: Path) -> None:
    """Process every raw CSV found under ``src_root``."""
    prepare_output_directories(dst_root, VALID_STATES)

    for csv_path in src_root.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            state = infer_state(df, csv_path)
            if state is None:
                continue

            dst_path = dst_root / state / csv_path.name
            shutil.copy2(csv_path, dst_path)

            processed = process_house_data(df.copy())
            processed.to_csv(dst_path, index=False)
            print(f"[OK] {csv_path} -> {dst_path}")
        except Exception as exc:
            print(f"[ERROR] Failed to process {csv_path}: {exc}")


if __name__ == "__main__":
    process_tree(SRC_ROOT, DST_ROOT)
