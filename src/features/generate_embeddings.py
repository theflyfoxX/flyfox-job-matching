# src/features/generate_embeddings.py

from src.io.ingest import load_all_raw
from src.features.embed_text import generate_job_embeddings, generate_applicant_embeddings
import os

if __name__ == "__main__":
    data = load_all_raw()
    os.makedirs("embeddings", exist_ok=True)

    generate_job_embeddings(data["jobs"], "embeddings/job_embeddings.parquet")
    generate_applicant_embeddings(data["experience"], "embeddings/applicant_embeddings.parquet")
