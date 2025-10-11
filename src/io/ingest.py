import os
import pandas as pd
from src.utils import logging_util

# Resolve path to <project-root>/data/raw
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

FILES = {
    "jobs": "Combined_Jobs_Final.csv",
    "experience": "Experience.csv",
    "job_data": "job_data.csv",
    "views": "Job_Views.csv",
    "interests": "Positions_Of_Interest.csv",
}

EXPECTED_COLUMNS = {
    "jobs": {
        "Job.ID", "Title", "Position", "Company", "City",
        "State.Name", "State.Code", "Job.Description"
    },
    "experience": {
        "Applicant.ID", "Position.Name", "Employer.Name", "City",
        "State.Name", "State.Code", "Start.Date", "End.Date"
    },
    "job_data": {
        "Job.ID", "text"
    },
    "views": {
        "Applicant.ID", "Job.ID", "Title", "Company", "View.Start", "View.End"
    },
    "interests": {
        "Applicant.ID", "Position.Of.Interest", "Created.At"
    }
}


def load_csv(file_key: str) -> pd.DataFrame:
    """Load and validate a raw CSV file."""
    file_path = os.path.join(RAW_DIR, FILES[file_key])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[✗] File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', quoting=1, encoding='utf-8', engine='python')
    except Exception as e:
        raise RuntimeError(f"[✗] Failed to read {file_key}: {e}")

    expected = EXPECTED_COLUMNS[file_key]
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"[✗] {file_key}: Missing columns: {missing}")

    logging_util.log_info(f"[✓] Loaded {file_key}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_all_raw():
    """Load and validate all raw datasets."""
    return {key: load_csv(key) for key in FILES}


if __name__ == "__main__":
    datasets = load_all_raw()
    for k, v in datasets.items():
        print(f"{k}: {v.shape}")
