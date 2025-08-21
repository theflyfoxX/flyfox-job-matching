# src/prep/ground_truth.py

import pandas as pd
import random

from src.io.ingest import load_all_raw




def build_ground_truth(views_df: pd.DataFrame, interests_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build labeled (job_id, applicant_id, label) pairs.
    Label = 1 if there was a real view or interest.
    """
    print("[*] Creating positive samples...")

    positives_views = views_df[["Job.ID", "Applicant.ID"]].drop_duplicates()
    positives_views["label"] = 1

    # Merge in interests just to add more positive samples
    positives_interests = interests_df.rename(columns={"Position.Of.Interest": "Job.ID"})[
        ["Job.ID", "Applicant.ID"]
    ].drop_duplicates()
    positives_interests["label"] = 1

    # Combine both and deduplicate again
    positives = pd.concat([positives_views, positives_interests], ignore_index=True)
    positives = positives.drop_duplicates()

    print(f"[✓] Total positive samples: {len(positives)}")

    return positives

if __name__ == "__main__":

    data = load_all_raw()
    ground_truth = build_ground_truth(data["views"], data["interests"])
    ground_truth.to_csv("data/interim/labeled_applicant_job_pairs.csv", index=False)
    print("[✓] Saved: data/interim/labeled_applicant_job_pairs.csv")
