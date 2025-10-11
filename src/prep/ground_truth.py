import os
import pandas as pd

from src.io.ingest import load_all_raw
from src.utils import logging_util
# src/features/build_ground_truth.py
def build_ground_truth(views_df: pd.DataFrame, interests_df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    logging_util.log_info("[*] Creating positive samples...")

    # 1) True positives from views (already valid Job.IDs)
    positives_views = views_df[["Job.ID", "Applicant.ID"]].drop_duplicates()
    positives_views["label"] = 1

    # 2) Interests → map by exact Title to real Job.IDs (strict)
    jobs_titles = jobs_df[["Job.ID", "Title"]].dropna().copy()
    jobs_titles["Title_norm"] = jobs_titles["Title"].astype(str).str.strip().str.lower()

    ints = interests_df[["Applicant.ID", "Position.Of.Interest"]].dropna().copy()
    ints["Title_norm"] = ints["Position.Of.Interest"].astype(str).str.strip().str.lower()

    # exact title match to assign a real Job.ID
    ints_join = ints.merge(jobs_titles, on="Title_norm", how="inner")
    positives_interests = ints_join[["Job.ID", "Applicant.ID"]].drop_duplicates()
    positives_interests["label"] = 1

    positives = pd.concat([positives_views, positives_interests], ignore_index=True).drop_duplicates()

    logging_util.log_info(f"[✓] Total positive samples: {len(positives)} "
                          f"(views={len(positives_views)}, interests-mapped={len(positives_interests)})")
    return positives

if __name__ == "__main__":
    # Get root-relative output path
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
    OUTPUT_PATH = os.path.join(INTERIM_DIR, "labeled_applicant_job_pairs.csv")

    # Load raw data
    data = load_all_raw()

    # Build and save ground truth
    ground_truth = build_ground_truth(data["views"], data["interests"])

    os.makedirs(INTERIM_DIR, exist_ok=True)
    ground_truth.to_csv(OUTPUT_PATH, index=False)
    logging_util.log_info(f"[✓] Saved: {OUTPUT_PATH}")
