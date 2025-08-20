# features/generate_applicant_embeddings_from_labeled.py

import os
import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELED_PATH = os.path.join(PROJ_ROOT, "data", "labeled_applicant_job_pairs.csv")
SAVE_PATH = os.path.join(PROJ_ROOT, "embeddings", "applicants", "embeddings_dict.npy")

def main():
    logging.info("Loading labeled data...")
    df = pd.read_csv(LABELED_PATH)

    logging.info("Extracting unique applicant IDs...")
    df = df[["Applicant.ID"]].drop_duplicates()
    df["Applicant.ID"] = df["Applicant.ID"].astype(str)
    df["text"] = df["Applicant.ID"].apply(lambda x: f"applicant {x}")

    applicant_ids = df["Applicant.ID"].tolist()
    applicant_texts = df["text"].tolist()

    logging.info(f"Generating embeddings for {len(applicant_ids)} applicants...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(applicant_texts, show_progress_bar=True, convert_to_numpy=True)

    logging.info("Saving applicant embeddings...")
    embeddings_dict = dict(zip(applicant_ids, embeddings))
    np.save(SAVE_PATH, embeddings_dict)
    logging.info(f"Applicant embeddings saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
