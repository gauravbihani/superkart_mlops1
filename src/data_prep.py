"""Step 2 - Data preparation.

Loads the raw CSV straight from the repository data folder, cleans it, drops
columns that should not be model inputs, splits into train/test and saves the
splits to data/processed/. The processed folder is what the GitHub Actions
workflow uploads as an artifact for the training job.

Usage (from the repo root):
    python src/data_prep.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/SuperKart.csv")
OUT_DIR = Path("data/processed")
TARGET = "Product_Store_Sales_Total"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove exact duplicates (none expected, but keeps the step safe to re-run)
    df = df.drop_duplicates()

    # 2. Fix inconsistent labels: 'reg' and 'Regular' are the same sugar category
    df["Product_Sugar_Content"] = (
        df["Product_Sugar_Content"].str.strip().replace({"reg": "Regular"})
    )

    # 3. Product_Id starts with two letters that mark the broad product family
    #    (FD = food, DR = drink, NC = non-consumable). Keep that, drop the id.
    df["Product_Family"] = df["Product_Id"].str[:2]

    # 4. Drop unnecessary columns:
    #    - Product_Id : unique per row, no predictive value once family is extracted
    #    - Store_Id   : fully determined by Store_Size / City_Type / Store_Type /
    #                   Establishment_Year, so it only duplicates information
    df = df.drop(columns=["Product_Id", "Store_Id"])

    # 5. Drop rows with missing values (the dataset has none, guard for new data)
    df = df.dropna()
    return df


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw data: {df.shape}")

    df = clean(df)
    print(f"Cleaned data   : {df.shape}")
    print("Sugar labels   :", sorted(df["Product_Sugar_Content"].unique()))

    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"Saved {OUT_DIR / 'train.csv'} {train.shape} and {OUT_DIR / 'test.csv'} {test.shape}")


if __name__ == "__main__":
    main()
