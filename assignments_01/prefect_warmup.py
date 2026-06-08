"""
Python 200: Assignment 01 - Prefect Warmup
Author: Ben Chapman
"""

import pandas as pd
import numpy as np
from prefect import task, flow

# --- Prefect Pipeline ---

@task(name="Create Pandas Series")
def create_series(arr):
    """return pandas Series named 'values'."""
    return pd.Series(arr, name="values")

@task(name="Clean Data (Drop NaNs)")
def clean_data(series):
    """Remove NaN values from series"""
    return series.dropna()

@task(name="Summarize Statistics")
def summarize_data(series):
    """Return dictionary of summary stats"""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow(name="Data Summary Pipeline Flow")
def pipeline_flow():
    """Chainsteps together"""
    # Define
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    
    # Step 1: Ingest and format
    raw_series = create_series(arr)
    
    # Step 2: Clean and transform
    cleaned_series = clean_data(raw_series)
    
    # Step 3: Analyze and output
    summary_dict = summarize_data(cleaned_series)
    
    return summary_dict

if __name__ == "__main__":
    final_summary = pipeline_flow()
    
    print("\n--- Pipeline Results ---")
    for key, value in final_summary.items():
        print(f"{key.capitalize()}: {value:.4f}")

# ==========================================
# --- CONCEPTUAL QUESTIONS ---
#
# Q: Why might Prefect be more overhead than it is worth here?
# A: Setting up the orchestration takes 
# vastly more time and compute power than the actual data processing since script posseses jsut 12 numbers!
#
# Q: Describe some realistic scenarios where a framework like Prefect could 
# still be useful, even if the pipeline logic itself stays simple.
# A: 
# 1. Scheduling: If this simple calculation needed to run automatically every 
#    morning at 6 AM without human intervention.
# 2. Retries: If downloading the array from an 
#    external API that occasionally timed out, Prefect could automatically 
#    retry the task 3 times before failing.
# 3. Observability & Alerts: If you have dozens of small pipelines running, 
#    Prefect gives you a centralized dashboard to see which ones passed/failed 
#    and can automatically send a Slack or email alert when things break.
# ==========================================