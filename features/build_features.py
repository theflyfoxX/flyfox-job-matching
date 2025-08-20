import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --------------------
# Config
# --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "features.csv")

LABELED_PATH = os.path.join(DATA_DIR, "labeled_applicant_job_pairs.csv")
EXPERIENCE_PATH = os.path.join(PROJ_ROOT, "Experience.csv")
INTEREST_PATH = os.path.join(PROJ_ROOT, "Positions_Of_Interest.csv")

JOB_EMBED_PATH = os.path.join(PROJ_ROOT, "embeddings", "job_titles", "embeddings_dict.npy")
APP_EMBED_PATH = os.path.join(PROJ_ROOT, "embeddings", "applicants", "embeddings_dict.npy")

# --------------------
# Load experience
# --------------------
def load_experience(path):
    df = pd.read_csv(path)

    df["Position.Name"] = df["Position.Name"].fillna("")
    df["Start.Date"] = pd.to_datetime(df["Start.Date"], errors="coerce")
    df["End.Date"] = pd.to_datetime(df["End.Date"], errors="coerce")
    now = pd.Timestamp.now()
    df["End.Date"] = df["End.Date"].fillna(now)

    df["years"] = (df["End.Date"] - df["Start.Date"]).dt.days / 365.25
    df["years"] = df["years"].clip(lower=0)

    # Aggregate
    agg = (
        df.groupby("Applicant.ID")
        .agg(
            exp_years_total=("years", "sum"),
            exp_last_city=("City", "last"),
            exp_last_state=("State.Code", "last"),
            exp_recency_days=("End.Date", lambda d: (now - d.max()).days)
        )
        .reset_index()
    )

    return agg

# --------------------
# Load interests
# --------------------
def load_interests(path):
    df = pd.read_csv(path)
    df = (
        df.groupby("Applicant.ID")["Position.Of.Interest"]
        .apply(lambda s: " ".join(s.dropna().unique()))
        .reset_index()
    )
    return df

# --------------------
# Cosine similarity
# --------------------
def compute_embedding_similarity(df, job_embeddings, app_embeddings):
    sims = []
    missing_job, missing_app = 0, 0

    for _, row in df.iterrows():
        job_id = str(row["Job.ID"]).strip()
        app_id = str(row["Applicant.ID"]).strip()

        job_vec = job_embeddings.get(job_id)
        app_vec = app_embeddings.get(app_id)

        if job_vec is None:
            missing_job += 1
        if app_vec is None:
            missing_app += 1

        if job_vec is None or app_vec is None:
            sims.append(0.0)
        else:
            sims.append(float(cosine_similarity([job_vec], [app_vec])[0][0]))

    df["embedding_similarity"] = sims
    logging.warning(f"Missing job embeddings: {missing_job}")
    logging.warning(f"Missing applicant embeddings: {missing_app}")
    return df

# --------------------
# Add structured features
# --------------------
def add_structured_features(df):
    df["state_match"] = (
        df["State.Code"].fillna("").str.lower() == df["exp_last_state"].fillna("").str.lower()
    ).astype(int)

    df["city_match"] = (
        df["City"].fillna("").str.lower() == df["exp_last_city"].fillna("").str.lower()
    ).astype(int)

    # Fallback: location_match based on text content (e.g. exp_last_city in job title)
    df["location_match"] = df.apply(
        lambda row: str(row.get("exp_last_city", "")).lower() in str(row.get("text", "")).lower(),
        axis=1
    ).astype(int)

    # Optionally create dummy placeholders if needed for your model
    df["industry_match"] = df["location_match"]  # or better: NLP similarity on job description
    df["position_match"] = df["location_match"]  # or better: NLP similarity on job title vs interest

    df["exp_recency_days"] = df["exp_recency_days"].fillna(9999)
    df["exp_years_total"] = df["exp_years_total"].fillna(0)

    return df

# --------------------
# Main
# --------------------
def main():
    logging.info("Loading base labeled pairs...")
    base = pd.read_csv(LABELED_PATH)

    logging.info("Loading experience and interest data...")
    exp_df = load_experience(EXPERIENCE_PATH)
    interest_df = load_interests(INTEREST_PATH)

    logging.info("Merging features...")
    df = base.merge(exp_df, on="Applicant.ID", how="left")
    df = df.merge(interest_df, on="Applicant.ID", how="left")

    logging.info("Loading embeddings...")
    job_embeddings = np.load(JOB_EMBED_PATH, allow_pickle=True).item()
    app_embeddings = np.load(APP_EMBED_PATH, allow_pickle=True).item()

    logging.info("Computing embedding similarity...")
    df = compute_embedding_similarity(df, job_embeddings, app_embeddings)

    logging.info("Adding structured features...")
    df = add_structured_features(df)

    logging.info("Saving final feature set...")
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Features saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
