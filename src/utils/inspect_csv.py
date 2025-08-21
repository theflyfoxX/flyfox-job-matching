# src/utils/inspect_csv.py
import os
import pandas as pd
from src.utils import logging_util

def inspect_csv(file_path: str, nrows: int = 5):
    """Print the columns and first few rows of a CSV file in a clean format."""
    if not os.path.exists(file_path):
        logging_util.log_error(f" File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path, nrows=nrows)

        logging_util.log_info("\n" + "=" * 80)
        logging_util.log_info(f"File: {file_path}")
        logging_util.log_info("-" * 80)

        logging_util.log_info(" Columns:")
        for idx, col in enumerate(df.columns, start=1):
            logging_util.log_info(f"   {idx}. {col}")

        logging_util.log_info("-" * 80)
        logging_util.log_info(f" Preview of first {nrows} rows:")
        logging_util.log_info(df.head().to_string(index=False))  # prevent messy wrapping
        logging_util.log_info("=" * 80)

    except Exception as e:
        logging_util.log_error(f" Could not read {file_path}: {e}")


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
