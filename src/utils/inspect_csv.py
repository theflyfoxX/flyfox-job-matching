import logging
import os
import pandas as pd

# Define absolute project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
logger = logging.getLogger(__name__)
def inspect_csv(file_path: str, nrows: int = 5):
    """Print the columns and first few rows of a CSV file in a clean format."""
    if not os.path.exists(file_path):
        logger.info(f" File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path, nrows=nrows)

        logger.info("\n" + "=" * 80)
        logger.info(f"File: {file_path}")
        logger.info("-" * 80)

        logger.info(" Columns:")
        for idx, col in enumerate(df.columns, start=1):
           logger.info(f"   {idx}. {col}")

           logger.info("-" * 80)
           logger.info(f" Preview of first {nrows} rows:")
           logger.info(df.head().to_string(index=False))
           logger.info("=" * 80)

    except Exception as e:
        logger.warning(f" Could not read {file_path}: {e}")


if __name__ == "__main__":
    files_to_check = [
        "Combined_Jobs_Final.csv",
        "Experience.csv",
        "Positions_Of_Interest.csv",
        "Job_Views.csv",
        "job_data.csv"
    ]

    for fname in files_to_check:
        full_path = os.path.join(RAW_DIR, fname)
        inspect_csv(full_path)
