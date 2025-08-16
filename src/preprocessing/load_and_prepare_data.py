# src/preprocessing/load_and_prepare_data.py
import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
FILES = {
    "jobs": os.path.join(DATA_DIR, "Combined_Jobs_Final.csv"),
    "experience": os.path.join(DATA_DIR, "Experience.csv"),
    "views": os.path.join(DATA_DIR, "Job_Views.csv"),
    "interests": os.path.join(DATA_DIR, "Positions_Of_Interest.csv"),
    "job_data": os.path.join(DATA_DIR, "job_data.csv"),
}

def load_csv(name, path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load {name} from {path}: {e}")
        return pd.DataFrame()

def inspect_df(df, name):
    logging.info(f"Inspecting {name}:")
    logging.info(f"  Nulls per column:\n{df.isnull().sum()}")
    logging.info(f"  Duplicates: {df.duplicated().sum()}")
    logging.info(f"  Data types:\n{df.dtypes}")

def label_matches(views_df, jobs_df, max_negatives_per_applicant=5):
    # Positive matches (applicant viewed the job)
    positive = views_df[['Applicant.ID', 'Job.ID']].drop_duplicates()
    positive['match'] = 1

    # Create set of all jobs
    all_jobs = jobs_df['Job.ID'].unique()

    # Generate negatives by sampling random jobs NOT viewed
    negatives = []

    for applicant_id, group in views_df.groupby('Applicant.ID'):
        viewed_jobs = set(group['Job.ID'])
        unviewed_jobs = list(set(all_jobs) - viewed_jobs)
        
        sampled_jobs = np.random.choice(unviewed_jobs, size=min(max_negatives_per_applicant, len(unviewed_jobs)), replace=False)
        
        for job_id in sampled_jobs:
            negatives.append((applicant_id, job_id, 0))

    df_neg = pd.DataFrame(negatives, columns=['Applicant.ID', 'Job.ID', 'match'])

    # Combine
    df_labeled = pd.concat([positive, df_neg], ignore_index=True)
    return df_labeled


def main():
    # Load
    df_jobs = load_csv("Combined_Jobs", FILES["jobs"])
    df_experience = load_csv("Experience", FILES["experience"])
    df_views = load_csv("Job_Views", FILES["views"])
    df_interests = load_csv("Positions_Of_Interest", FILES["interests"])
    df_job_text = load_csv("Job_Data", FILES["job_data"])
    
    df_job_text = df_job_text.drop(columns=['Unnamed: 0'], errors='ignore')

    # Inspect
    inspect_df(df_jobs, "Jobs")
    inspect_df(df_experience, "Experience")
    inspect_df(df_views, "Views")
    inspect_df(df_interests, "Interests")
    inspect_df(df_job_text, "Job Text")

    # Merge views with jobs and text
    df_merged = df_views.merge(df_jobs, on='Job.ID', how='left').merge(df_job_text, on='Job.ID', how='left')

    # Create match labels (based on views as positive interactions)
    df_labeled = label_matches(df_views, df_jobs)

    # Merge with job + text + experience + interest info
    df_final = df_labeled.merge(df_jobs, on='Job.ID', how='left')
    df_final = df_final.merge(df_job_text, on='Job.ID', how='left')
    df_final = df_final.merge(df_interests, on='Applicant.ID', how='left')

    output_path = os.path.join(DATA_DIR, "data", "labeled_applicant_job_pairs.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    logging.info(f"Labeled dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
