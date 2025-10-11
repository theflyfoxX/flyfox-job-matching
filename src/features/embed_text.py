import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

from src.utils import logging_util


def embed_texts(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def generate_job_embeddings(jobs_df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    logging_util.log_info("[*] Generating job embeddings...")
    texts = (jobs_df["Title"].fillna("") + " " + jobs_df["Job.Description"].fillna("")).tolist()
    embeddings = embed_texts(texts)
    job_emb_df = pd.DataFrame(embeddings)
    job_emb_df.insert(0, "Job.ID", jobs_df["Job.ID"].values)
    job_emb_df.to_parquet(save_path, index=False)
    logging_util.log_info(f"[✓] Saved job embeddings to {save_path}")
    return job_emb_df


def generate_applicant_embeddings(exp_df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    logging_util.log_info("[*] Generating applicant embeddings...")
    latest_exp = (
        exp_df.sort_values("End.Date", ascending=False)
        .drop_duplicates("Applicant.ID", keep="first")
        .copy()
    )

    texts = (
        latest_exp["Position.Name"].fillna("") + " " +
        latest_exp["Job.Description"].fillna("")
    ).tolist()

    embeddings = embed_texts(texts)
    applicant_emb_df = pd.DataFrame(embeddings)
    applicant_emb_df.insert(0, "Applicant.ID", latest_exp["Applicant.ID"].values)
    applicant_emb_df.to_parquet(save_path, index=False)
    logging_util.log_info(f"[✓] Saved applicant embeddings to {save_path}")
    return applicant_emb_df
