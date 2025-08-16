# src/features/build_features.py

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'labeled_applicant_job_pairs.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'features.csv')


def create_location_match(df):
    if 'City' in df.columns and 'City_y' in df.columns:
        df['location_match'] = (
            df['City'].str.lower().fillna('') == df['City_y'].str.lower().fillna('')
        ).astype(int)
    else:
        logging.warning("Missing 'City' columns for location_match")
        df['location_match'] = 0
    return df


def create_position_interest_match(df):
    if 'Position' in df.columns and 'Position.Of.Interest' in df.columns:
        df['position_interest_match'] = (
            df['Position'].str.lower().fillna('') == df['Position.Of.Interest'].str.lower().fillna('')
        ).astype(int)
    else:
        logging.warning("Missing 'Position' columns for position_interest_match")
        df['position_interest_match'] = 0
    return df


def main():
    logging.info(f"Loading labeled dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    logging.info("Creating location match feature...")
    df = create_location_match(df)

    logging.info("Creating position interest match feature...")
    df = create_position_interest_match(df)

    logging.info(f"Saving feature-enhanced dataset to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info("Done!")


if __name__ == "__main__":
    main()
