import os
from src.io.ingest import load_all_raw
from src.features.embed_text import generate_job_embeddings, generate_applicant_embeddings

# Define project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Embedding output paths
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
JOBS_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "jobs", "job_embeddings.parquet")
APPLICANTS_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "applicants", "applicant_embeddings.parquet")

if __name__ == "__main__":
    # Load raw data
    data = load_all_raw()

    # Create embedding folders if not exist
    os.makedirs(os.path.dirname(JOBS_EMB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(APPLICANTS_EMB_PATH), exist_ok=True)

    # Generate and save embeddings
    generate_job_embeddings(data["jobs"], JOBS_EMB_PATH)
    generate_applicant_embeddings(data["experience"], APPLICANTS_EMB_PATH)
