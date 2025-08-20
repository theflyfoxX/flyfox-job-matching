import pandas as pd
import numpy as np
import joblib
import logging
from features.build_features import compute_embedding_similarity, add_structured_features
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Loading trained model...")
    model = joblib.load("xgboost_model.pkl")

    logging.info("Loading new applicant–job pairs...")
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    pairs = pd.read_csv(os.path.join(DATA_DIR, "unlabeled_applicant_job_pairs.csv"))


    # Load experience and job data
    logging.info("Merging experience and interests...")
    exp = pd.read_csv("Experience.csv")
    job_data = pd.read_csv("job_data.csv")

    # Extract latest experience per applicant
    exp_latest = (
        exp.sort_values("Created.At")
        .drop_duplicates("Applicant.ID", keep="last")
        .rename(columns={"City": "exp_last_city", "State.Code": "exp_last_state"})
    )

    # Merge metadata into the pairs dataframe
    df = pairs.copy()
    df = df.merge(job_data, on="Job.ID", how="left")
    df = df.merge(exp_latest[["Applicant.ID", "exp_last_city", "exp_last_state"]], on="Applicant.ID", how="left")

    # Load embeddings
    logging.info("Loading embeddings...")
    job_embeddings = np.load("embeddings/jobs/embeddings_dict.npy", allow_pickle=True).item()
    applicant_embeddings = np.load("embeddings/applicants/embeddings_dict.npy", allow_pickle=True).item()

    # Compute embedding similarity
    logging.info("Computing embedding similarity...")
    df = compute_embedding_similarity(df, job_embeddings, applicant_embeddings)

    # Optional: filter out missing embeddings
    missing_jobs = ~df["Job.ID"].isin(job_embeddings.keys())
    missing_applicants = ~df["Applicant.ID"].isin(applicant_embeddings.keys())
    logging.warning(f"Missing job embeddings: {missing_jobs.sum()}")
    logging.warning(f"Missing applicant embeddings: {missing_applicants.sum()}")
    df = df[~(missing_jobs | missing_applicants)]

    # Add structured features
    logging.info("Adding structured features...")
    df = add_structured_features(df)

    # Select only feature columns used during training
    feature_cols = ['embedding_similarity', 'location_match', 'exp_years_total', 'exp_recency_days']
    X = df[feature_cols]

    logging.info("Predicting match probabilities...")
    df["match_probability"] = model.predict_proba(X)[:, 1]

    # Save predictions
    output_path = "predictions.csv"
    df[["Applicant.ID", "Job.ID", "match_probability"]].to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
