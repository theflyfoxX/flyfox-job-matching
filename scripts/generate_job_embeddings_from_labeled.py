# features/generate_job_embeddings_from_labeled.py

import os
import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELED_PATH = os.path.join(PROJ_ROOT, "data", "labeled_applicant_job_pairs.csv")
SAVE_PATH = os.path.join(PROJ_ROOT, "embeddings", "job_titles", "embeddings_dict.npy")

def main():
    logging.info("Loading labeled data...")
    df = pd.read_csv(LABELED_PATH)

    logging.info("Preparing job ID and text pairs...")
    df = df[["Job.ID", "text"]].dropna().drop_duplicates()
    df["Job.ID"] = df["Job.ID"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    job_texts = df["text"].tolist()
    job_ids = df["Job.ID"].tolist()

    logging.info(f"Generating embeddings for {len(job_texts)} job descriptions...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(job_texts, show_progress_bar=True, convert_to_numpy=True)

    logging.info("Saving job embeddings...")
    embeddings_dict = dict(zip(job_ids, embeddings))
    np.save(SAVE_PATH, embeddings_dict)
    logging.info(f"Job embeddings saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
