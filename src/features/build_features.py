import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Config ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

LABELED_PATH = os.path.join(INTERIM_DIR, "labeled_applicant_job_pairs.csv")
EXPERIENCE_PATH = os.path.join(RAW_DIR, "Experience.csv")
INTEREST_PATH = os.path.join(RAW_DIR, "Positions_Of_Interest.csv")
JOBS_PATH = os.path.join(RAW_DIR, "Combined_Jobs_Final.csv")

# Load embeddings from PARQUET (authoritative)
JOB_EMBED_PARQUET = os.path.join(PROJECT_ROOT, "embeddings", "jobs", "job_embeddings.parquet")
APP_EMBED_PARQUET = os.path.join(PROJECT_ROOT, "embeddings", "applicants", "applicant_embeddings.parquet")

OUTPUT_PATH = os.path.join(FEATURES_DIR, "features.csv")

# ------------------ Helpers ------------------

def _parquet_to_dict(path: str, id_col: str) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    if id_col not in df.columns:
        raise KeyError(f"Expected '{id_col}' in {path}, got: {list(df.columns)[:12]}")
    ids = df[id_col].astype(str).str.strip()
    vec_cols = [c for c in df.columns if c != id_col]
    # keep a deterministic column order
    vec_cols = sorted(vec_cols, key=lambda x: (len(str(x)), str(x)))
    mat = df[vec_cols].to_numpy(dtype=float)
    return {i: v for i, v in zip(ids, mat)}

# ------------------ Features ------------------

def compute_embedding_similarity(df, job_embeddings, app_embeddings, drop_missing=False):
    # normalize keys defensively
    job_embeddings = {str(k).strip(): np.asarray(v) for k, v in job_embeddings.items()}
    app_embeddings = {str(k).strip(): np.asarray(v) for k, v in app_embeddings.items()}

    j_ids = df["Job.ID"].astype(str).str.strip().tolist()
    a_ids = df["Applicant.ID"].astype(str).str.strip().tolist()

    has_job = [jid in job_embeddings for jid in j_ids]
    has_app = [aid in app_embeddings for aid in a_ids]
    has_both = np.array(has_job) & np.array(has_app)

    total = len(df)
    miss_jobs = total - int(np.sum(has_job))
    miss_apps = total - int(np.sum(has_app))

    # diagnostics files
    os.makedirs(FEATURES_DIR, exist_ok=True)
    missing_jobs = sorted({jid for jid, ok in zip(j_ids, has_job) if not ok})
    missing_apps = sorted({aid for aid, ok in zip(a_ids, has_app) if not ok})
    pd.Series(missing_jobs, name="Job.ID").to_csv(os.path.join(FEATURES_DIR, "missing_job_embeddings.csv"), index=False)
    pd.Series(missing_apps, name="Applicant.ID").to_csv(os.path.join(FEATURES_DIR, "missing_app_embeddings.csv"), index=False)

    pct_jobs = 100 * (miss_jobs / total) if total else 0.0
    pct_apps = 100 * (miss_apps / total) if total else 0.0
    logging.warning(f"Missing job embeddings: {miss_jobs} ({pct_jobs:.1f}%)")
    logging.warning(f"Missing applicant embeddings: {miss_apps} ({pct_apps:.1f}%)")

    if drop_missing:
        df = df.loc[has_both].copy()
        logging.info(f"Dropped rows without both embeddings. New size: {len(df)}")
        j_ids = df["Job.ID"].astype(str).str.strip().tolist()
        a_ids = df["Applicant.ID"].astype(str).str.strip().tolist()

    sims, flags = [], []
    for jid, aid in zip(j_ids, a_ids):
        jv = job_embeddings.get(jid)
        av = app_embeddings.get(aid)
        if jv is None or av is None:
            sims.append(0.0)
            flags.append(0)
        else:
            sims.append(float(cosine_similarity([jv], [av])[0][0]))
            flags.append(1)

    df["embedding_similarity"] = sims
    df["has_both_embeds"] = flags
    return df

def add_structured_features(df):
    job_state = df.get("State.Code", pd.Series([""] * len(df))).fillna("").str.lower()
    job_city  = df.get("City", pd.Series([""] * len(df))).fillna("").str.lower()
    text_col  = df.get("text", pd.Series([""] * len(df))).fillna("").str.lower()

    exp_state = df.get("exp_last_state", pd.Series([""] * len(df))).fillna("").str.lower()
    exp_city  = df.get("exp_last_city", pd.Series([""] * len(df))).fillna("").str.lower()

    df["state_match"] = (job_state == exp_state).astype(int)
    df["city_match"]  = (job_city == exp_city).astype(int)
    df["location_match"] = df.apply(
        lambda r: (str(r.get("exp_last_city", "")).lower() in str(r.get("text", "")).lower())
                  if pd.notna(r.get("exp_last_city", "")) else 0,
        axis=1
    ).astype(int)
    df["industry_match"] = df["location_match"]
    df["position_match"] = df["location_match"]

    df["exp_recency_days"] = df.get("exp_recency_days", pd.Series([9999]*len(df))).fillna(9999)
    df["exp_years_total"] = df.get("exp_years_total", pd.Series([0]*len(df))).fillna(0)
    return df

# ------------------ Data loading ------------------

def load_experience(path):
    df = pd.read_csv(path)
    df["Applicant.ID"] = df["Applicant.ID"].astype(str).str.strip()

    df["Position.Name"] = df["Position.Name"].fillna("")
    df["Start.Date"] = pd.to_datetime(df["Start.Date"], errors="coerce")
    df["End.Date"]   = pd.to_datetime(df["End.Date"], errors="coerce")

    now = pd.Timestamp.now()
    df["End.Date"] = df["End.Date"].fillna(now)
    df["years"] = (df["End.Date"] - df["Start.Date"]).dt.days / 365.25
    df["years"] = df["years"].clip(lower=0)

    agg = (df.groupby("Applicant.ID")
             .agg(exp_years_total=("years","sum"),
                  exp_last_city=("City","last"),
                  exp_last_state=("State.Code","last"),
                  exp_recency_days=("End.Date", lambda d: (now - d.max()).days))
             .reset_index())
    return agg

def load_interests(path):
    df = pd.read_csv(path)
    df["Applicant.ID"] = df["Applicant.ID"].astype(str).str.strip()
    return (df.groupby("Applicant.ID")["Position.Of.Interest"]
              .apply(lambda s: " ".join(s.dropna().unique()))
              .reset_index())

# ------------------ Main ------------------

def main():
    logging.info("Loading base labeled pairs...")
    base = pd.read_csv(LABELED_PATH)
    base["Job.ID"] = base["Job.ID"].astype(str).str.strip()
    base["Applicant.ID"] = base["Applicant.ID"].astype(str).str.strip()

    logging.info("Loading experience & interests...")
    exp_df = load_experience(EXPERIENCE_PATH)
    interest_df = load_interests(INTEREST_PATH)

    logging.info("Loading jobs (strict schema)...")
    required_cols = [
        "Job.ID", "City", "State.Name", "State.Code",
        "Title", "Position", "Industry",
        "Job.Description", "Requirements"
    ]
    jobs_df = pd.read_csv(
        JOBS_PATH,
        engine="python",
        usecols=required_cols,
        on_bad_lines="skip",
        na_values=["NA"]
    )

    # strict validation
    missing = [c for c in required_cols if c not in jobs_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in Combined_Jobs_Final.csv: {missing}")

    # deterministic text field
    for c in ["Title", "Position", "Industry", "Job.Description", "Requirements"]:
        jobs_df[c] = jobs_df[c].fillna("").astype(str)
    jobs_df["Job.ID"] = jobs_df["Job.ID"].astype(str).str.strip()
    jobs_df["text"] = (
        jobs_df["Title"] + " " +
        jobs_df["Position"] + " " +
        jobs_df["Industry"] + " " +
        jobs_df["Job.Description"] + " " +
        jobs_df["Requirements"]
    ).str.strip()

    logging.info("Merging features...")
    df = base.merge(exp_df, on="Applicant.ID", how="left")
    df = df.merge(interest_df, on="Applicant.ID", how="left")
    df = df.merge(jobs_df[["Job.ID", "City", "State.Code", "text"]], on="Job.ID", how="left")

    logging.info("Loading embeddings (from Parquet)...")
    job_embeddings = _parquet_to_dict(JOB_EMBED_PARQUET, id_col="Job.ID")
    app_embeddings = _parquet_to_dict(APP_EMBED_PARQUET, id_col="Applicant.ID")

    logging.info("Computing embedding similarity...")
    df = compute_embedding_similarity(df, job_embeddings, app_embeddings, drop_missing=False)

    logging.info("Adding structured features...")
    df = add_structured_features(df)

    logging.info("Saving final feature set...")
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Features saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
