# src/prep/negative_sampling.py

import pandas as pd
import random
from typing import Set
from src.utils import logging_util


def generate_negatives(
    jobs_df: pd.DataFrame,
    applicants_df: pd.DataFrame,
    positives_df: pd.DataFrame,
    neg_per_pos: int = 3
) -> pd.DataFrame:
    logging_util.log_info("[*] Generating negative samples...")

    job_ids = jobs_df["Job.ID"].unique()
    applicant_ids = applicants_df["Applicant.ID"].unique()
    existing_pairs: Set[tuple] = set(map(tuple, positives_df[["Job.ID", "Applicant.ID"]].values))

    negatives = set()

    target_neg_count = len(positives_df) * neg_per_pos
    attempts = 0
    max_attempts = target_neg_count * 10  # safety

    while len(negatives) < target_neg_count and attempts < max_attempts:
        j = random.choice(job_ids)
        a = random.choice(applicant_ids)
        if (j, a) not in existing_pairs:
            negatives.add((j, a))
        attempts += 1

    logging_util.log_info(f"[✓] Generated {len(negatives)} negative samples.")

    df_neg = pd.DataFrame(list(negatives), columns=["Job.ID", "Applicant.ID"])
    df_neg["label"] = 0
    return df_neg

if __name__ == "__main__":
    from src.io.ingest import load_all_raw

    # Load raw data
    data = load_all_raw()

    # Load positives from ground_truth
    positives = pd.read_csv("data/interim/labeled_applicant_job_pairs.csv")

    # Generate negatives
    negatives = generate_negatives(
        jobs_df=data["jobs"],
        applicants_df=data["experience"],
        positives_df=positives,
        neg_per_pos=3  # can tune later
    )

    # Combine and save
    full_df = pd.concat([positives, negatives], ignore_index=True)
    full_df.to_csv("data/interim/labeled_applicant_job_pairs.csv", index=False)
    logging_util.log_info("[✓] Combined labeled dataset saved: data/interim/labeled_applicant_job_pairs.csv")
