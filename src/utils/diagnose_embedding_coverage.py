# tools/diagnose_embedding_coverage.py
import os, pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
EMB = os.path.join(PROJECT_ROOT, "embeddings")

pairs = pd.read_csv(os.path.join(INTERIM, "labeled_applicant_job_pairs.csv"))
pairs["Job.ID"] = pairs["Job.ID"].astype(str).str.strip()
pairs["Applicant.ID"] = pairs["Applicant.ID"].astype(str).str.strip()

jobs = pd.read_csv(os.path.join(RAW, "Combined_Jobs_Final.csv"), engine="python", on_bad_lines="skip",
                   usecols=["Job.ID","Title","Position","Industry","Job.Description","Requirements"])
jobs["Job.ID"] = jobs["Job.ID"].astype(str).str.strip()

exp = pd.read_csv(os.path.join(RAW, "Experience.csv"))
exp["Applicant.ID"] = exp["Applicant.ID"].astype(str).str.strip()

# load parquet embeddings
jobs_emb = pd.read_parquet(os.path.join(EMB, "jobs", "job_embeddings.parquet"))
apps_emb = pd.read_parquet(os.path.join(EMB, "applicants", "applicant_embeddings.parquet"))
jobs_emb["Job.ID"] = jobs_emb["Job.ID"].astype(str).str.strip()
apps_emb["Applicant.ID"] = apps_emb["Applicant.ID"].astype(str).str.strip()

want_job_ids = set(pairs["Job.ID"].unique())
have_job_ids = set(jobs_emb["Job.ID"].unique())
missing_job_ids = sorted(want_job_ids - have_job_ids)

want_app_ids = set(pairs["Applicant.ID"].unique())
have_app_ids = set(apps_emb["Applicant.ID"].unique())
missing_app_ids = sorted(want_app_ids - have_app_ids)

print(f"Jobs — need {len(want_job_ids)}, have {len(have_job_ids)}, missing {len(missing_job_ids)}")
print(f"Applicants — need {len(want_app_ids)}, have {len(have_app_ids)}, missing {len(missing_app_ids)}")

# Where do missing jobs come from?
in_raw_jobs = set(jobs["Job.ID"].astype(str).str.strip().unique())
missing_jobs_not_in_raw = sorted(set(missing_job_ids) - in_raw_jobs)
print(f"Missing jobs that are NOT in Combined_Jobs_Final.csv: {len(missing_jobs_not_in_raw)}")

# Which applicants have zero experience rows?
have_exp_app_ids = set(exp["Applicant.ID"].unique())
missing_apps_no_exp = sorted(set(missing_app_ids) - have_exp_app_ids)
print(f"Missing applicants with NO rows in Experience.csv: {len(missing_apps_no_exp)}")
