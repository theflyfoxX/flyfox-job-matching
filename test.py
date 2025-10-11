import pandas as pd
from src.utils import logging_util

df = pd.read_csv("job_data.csv")
logging_util.log_info(df.columns.tolist())
