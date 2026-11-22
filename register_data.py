"""Step 1 - Data registration.

Registers the raw SuperKart dataset: validates that every expected column is
present (with a sensible dtype), prints a short summary, and writes a small
registry record (data/registry.json) so later steps can see which dataset
version they are working on.

Usage (from the repo root):
    python src/register_data.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/SuperKart.csv")
REGISTRY_PATH = Path("data/registry.json")

EXPECTED_COLUMNS = {
    "Product_Id": "object",
    "Product_Weight": "number",
    "Product_Sugar_Content": "object",
    "Product_Allocated_Area": "number",
    "Product_Type": "object",
    "Product_MRP": "number",
    "Store_Id": "object",
    "Store_Establishment_Year": "number",
    "Store_Size": "object",
    "Store_Location_City_Type": "object",
    "Store_Type": "object",
    "Product_Store_Sales_Total": "number",
}


def validate(df: pd.DataFrame) -> list:
    """Return a list of problems found (empty list = dataset is valid)."""
    problems = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        problems.append(f"Missing columns: {missing}")
    if extra:
        problems.append(f"Unexpected columns: {extra}")

    for col, kind in EXPECTED_COLUMNS.items():
        if col not in df.columns:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if kind == "number" and not is_numeric:
            problems.append(f"Column {col} should be numeric, got {df[col].dtype}")
        if kind == "object" and is_numeric:
            problems.append(f"Column {col} should be text, got {df[col].dtype}")

    if len(df) == 0:
        problems.append("Dataset has no rows")
    return problems


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        return 1

    df = pd.read_csv(DATA_PATH)
    problems = validate(df)

    print("=" * 60)
    print("DATASET REGISTRATION - SuperKart")
    print("=" * 60)
    print(f"File           : {DATA_PATH}")
    print(f"Rows x columns : {df.shape[0]} x {df.shape[1]}")
    print(f"Missing values : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows : {int(df.duplicated().sum())}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("\nTarget (Product_Store_Sales_Total) summary:")
    print(df["Product_Store_Sales_Total"].describe().round(2).to_string())

    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    record = {
        "dataset": "SuperKart",
        "path": str(DATA_PATH),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    }
    REGISTRY_PATH.write_text(json.dumps(record, indent=2))
    print(f"\nVALIDATION PASSED - all {len(EXPECTED_COLUMNS)} expected columns present.")
    print(f"Registered dataset (sha256 {record['sha256'][:12]}...) -> {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
