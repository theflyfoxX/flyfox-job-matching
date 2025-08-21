# src/utils/inspect_csv.py
import os
import pandas as pd

def inspect_csv(file_path: str, nrows: int = 5):
    """Print the columns and first few rows of a CSV file in a clean format."""
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path, nrows=nrows)

        print("\n" + "=" * 80)
        print(f"File: {file_path}")
        print("-" * 80)

        print(" Columns:")
        for idx, col in enumerate(df.columns, start=1):
            print(f"   {idx}. {col}")

        print("-" * 80)
        print(f" Preview of first {nrows} rows:")
        print(df.head().to_string(index=False))  # prevent messy wrapping
        print("=" * 80)

    except Exception as e:
        print(f" Could not read {file_path}: {e}")


if __name__ == "__main__":
    files_to_check = [
        "data/raw/Combined_Jobs_Final.csv",
        "data/raw/Experience.csv",
        "data/raw/Positions_Of_Interest.csv",
        "data/raw/Job_Views.csv",
        "data/raw/job_data.csv"
    ]

    for f in files_to_check:
        inspect_csv(f)
