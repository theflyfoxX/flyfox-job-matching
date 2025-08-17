import os
import re
import math
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


PROJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.join(PROJ_ROOT, "data")

LABELED_PATH = os.path.join(DATA_DIR, "labeled_applicant_job_pairs.csv")
EXPERIENCE_PATH = os.path.join(PROJ_ROOT, "Experience.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "features.csv")


# -----------------------
# Small utils
# -----------------------
STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or that the to with
""".split())

TOKEN_RE = re.compile(r"[A-Za-z]+")

def safe_lower(s):
    if pd.isna(s):
        return ""
    return str(s).lower()

def tokenize(text: str):
    tokens = [t for t in TOKEN_RE.findall(safe_lower(text)) if t not in STOPWORDS and len(t) > 2]
    return tokens

def jaccard_overlap(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def parse_date_safe(val):
    if pd.isna(val):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(val), fmt)
        except Exception:
            continue
    try:
        # last resort: pandas parser
        return pd.to_datetime(val, errors="coerce").to_pydatetime()
    except Exception:
        return None

def years_between(start, end):
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).days / 365.25)


# -----------------------
# Experience aggregation
# -----------------------
def load_and_aggregate_experience(experience_path: str) -> pd.DataFrame:
    """
    Returns one row per Applicant.ID with:
      - exp_titles_concat: concatenated past Position.Name values
      - exp_last_position: most recent position
      - exp_last_city, exp_last_state
      - exp_years_total: total years across jobs
      - exp_recency_days: days since last End.Date (or 0 if current)
    """
    if not os.path.exists(experience_path):
        logging.warning(f"Experience file not found: {experience_path}. Proceeding without experience features.")
        return pd.DataFrame(columns=[
            "Applicant.ID", "exp_titles_concat", "exp_last_position",
            "exp_last_city", "exp_last_state", "exp_years_total", "exp_recency_days"
        ])

    exp = pd.read_csv(experience_path)

    # normalize expected columns if casing differs
    colmap = {}
    for col in exp.columns:
        low = col.strip().lower()
        if low == "applicant.id": colmap[col] = "Applicant.ID"
        elif low == "position.name": colmap[col] = "Position.Name"
        elif low == "employer.name": colmap[col] = "Employer.Name"
        elif low == "city": colmap[col] = "City"
        elif low == "state.name": colmap[col] = "State.Name"
        elif low == "state.code": colmap[col] = "State.Code"
        elif low == "start.date": colmap[col] = "Start.Date"
        elif low == "end.date": colmap[col] = "End.Date"
        elif low == "job.description": colmap[col] = "Job.Description"
    exp = exp.rename(columns=colmap)

    # gentle defaults
    for must in ["Applicant.ID", "Position.Name", "City", "State.Code", "Start.Date", "End.Date"]:
        if must not in exp.columns:
            exp[must] = np.nan

    # parse dates
    exp["_start_dt"] = exp["Start.Date"].apply(parse_date_safe)
    exp["_end_dt"] = exp["End.Date"].apply(parse_date_safe)
    now = datetime.utcnow()
    exp["_end_dt_filled"] = exp["_end_dt"].apply(lambda d: d if d is not None else now)

    exp["_years"] = exp.apply(lambda r: years_between(r["_start_dt"], r["_end_dt_filled"]), axis=1)

    # most recent job = max of end_dt_filled (current treated as now)
    idx = exp.groupby("Applicant.ID")["_end_dt_filled"].idxmax()
    most_recent = exp.loc[idx, ["Applicant.ID", "Position.Name", "City", "State.Code", "_end_dt_filled"]].copy()
    most_recent = most_recent.rename(columns={
        "Position.Name": "exp_last_position",
        "City": "exp_last_city",
        "State.Code": "exp_last_state",
        "_end_dt_filled": "_last_end"
    })

    # total years per applicant
    yrs = exp.groupby("Applicant.ID")["_years"].sum().reset_index().rename(columns={"_years": "exp_years_total"})

    # recency in days since last end date (0 if current)
    recency = most_recent[["Applicant.ID", "_last_end"]].copy()
    recency["exp_recency_days"] = recency["_last_end"].apply(lambda d: (now - d).days if d is not None else 9999)
    recency = recency.drop(columns=["_last_end"])

    # concatenate unique past positions
    titles = (
        exp.groupby("Applicant.ID")["Position.Name"]
        .apply(lambda s: " ".join(sorted(set(str(x) for x in s.dropna()))))
        .reset_index()
        .rename(columns={"Position.Name": "exp_titles_concat"})
    )

    agg = most_recent.drop(columns=["_last_end"]).merge(yrs, on="Applicant.ID", how="outer") \
                     .merge(recency, on="Applicant.ID", how="outer") \
                     .merge(titles, on="Applicant.ID", how="outer")

    # fill basics
    agg["exp_years_total"] = agg["exp_years_total"].fillna(0.0)
    agg["exp_recency_days"] = agg["exp_recency_days"].fillna(9999)

    return agg


# -----------------------
# Feature builders
# -----------------------
def feature_location_matches(df: pd.DataFrame) -> pd.DataFrame:
    # job location columns (from Combined_Jobs_Final.csv)
    job_city = "City"
    job_state = "State.Code"

    # applicant location from experience aggregation
    app_city = "exp_last_city"
    app_state = "exp_last_state"

    for c in [job_city, job_state, app_city, app_state]:
        if c not in df.columns:
            df[c] = np.nan

    df["state_match"] = (
        df[job_state].astype(str).str.lower().fillna("") ==
        df[app_state].astype(str).str.lower().fillna("")
    ).astype(int)

    df["city_match"] = (
        df[job_city].astype(str).str.lower().fillna("") ==
        df[app_city].astype(str).str.lower().fillna("")
    ).astype(int)

    return df


def feature_interest_matches(df: pd.DataFrame) -> pd.DataFrame:
    # strict equality
    if "Position" not in df.columns:
        df["Position"] = ""
    if "Position.Of.Interest" not in df.columns:
        df["Position.Of.Interest"] = ""

    df["position_interest_match"] = (
        df["Position"].astype(str).str.lower().fillna("") ==
        df["Position.Of.Interest"].astype(str).str.lower().fillna("")
    ).astype(int)

    # token overlap (Jaccard) for fuzzier signal
    df["_pos_tokens"] = df["Position"].fillna("").map(tokenize)
    df["_poi_tokens"] = df["Position.Of.Interest"].fillna("").map(tokenize)
    df["position_interest_overlap"] = [
        jaccard_overlap(a, b) for a, b in zip(df["_pos_tokens"], df["_poi_tokens"])
    ]
    df = df.drop(columns=["_pos_tokens", "_poi_tokens"], errors="ignore")
    return df


def feature_title_similarity_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine similarity between job (Position + Title) and applicant's exp_titles_concat.
    """
    for c in ["Position", "Title", "exp_titles_concat"]:
        if c not in df.columns:
            df[c] = ""

    job_side = (df["Position"].fillna("") + " " + df["Title"].fillna("")).tolist()
    app_side = df["exp_titles_concat"].fillna("").tolist()

    # vectorize both sides in a single vocab for consistent space
    texts = job_side + app_side
    if len(texts) == 0:
        df["title_similarity_tfidf"] = 0.0
        return df

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    mat = tfidf.fit_transform(texts)

    n = len(job_side)
    job_mat = mat[:n]
    app_mat = mat[n:]

    # cosine similarity row-wise
    sims = []
    # ensure same row alignment
    for i in range(n):
        js = job_mat[i]
        as_ = app_mat[i]
        if js.nnz == 0 or as_.nnz == 0:
            sims.append(0.0)
        else:
            sims.append(float(cosine_similarity(js, as_)[0, 0]))

    df["title_similarity_tfidf"] = sims
    return df


def feature_desc_keyword_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keyword overlap between job description text and applicant positions.
    """
    if "text" not in df.columns:
        df["text"] = ""
    if "exp_titles_concat" not in df.columns:
        df["exp_titles_concat"] = ""

    job_tokens = df["text"].fillna("").map(tokenize)
    app_tokens = df["exp_titles_concat"].fillna("").map(tokenize)

    df["desc_keyword_overlap"] = [
        jaccard_overlap(a, b) for a, b in zip(job_tokens, app_tokens)
    ]
    return df


def feature_experience_numbers(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_years_total" not in df.columns:
        df["exp_years_total"] = 0.0
    if "exp_recency_days" not in df.columns:
        df["exp_recency_days"] = 9999.0

    # simple caps/normalizations to reduce outliers
    df["exp_years_total_capped"] = df["exp_years_total"].clip(0, 40)
    df["exp_recency_days_capped"] = df["exp_recency_days"].clip(0, 3650)  # ~10 years

    return df


# -----------------------
# Main
# -----------------------
def main():
    # 1) Load base labeled pairs
    if not os.path.exists(LABELED_PATH):
        raise FileNotFoundError(f"Base labeled dataset not found: {LABELED_PATH}")

    logging.info(f"Loading base labeled pairs: {LABELED_PATH}")
    base = pd.read_csv(LABELED_PATH)

    # 2) Load + aggregate experience (1 row per applicant)
    logging.info(f"Loading & aggregating experience: {EXPERIENCE_PATH}")
    exp_agg = load_and_aggregate_experience(EXPERIENCE_PATH)

    # 3) Merge experience aggregates into base on Applicant.ID
    df = base.merge(exp_agg, on="Applicant.ID", how="left")

    # 4) Build features
    logging.info("Building location match features...")
    df = feature_location_matches(df)

    logging.info("Building position/interest features...")
    df = feature_interest_matches(df)

    logging.info("Building TF-IDF title similarity...")
    df = feature_title_similarity_tfidf(df)

    logging.info("Building description keyword overlap...")
    df = feature_desc_keyword_overlap(df)

    logging.info("Building numeric experience features...")
    df = feature_experience_numbers(df)

    # 5) Save
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Feature dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
