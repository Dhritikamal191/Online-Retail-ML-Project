import os
import json
import joblib
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import streamlit as st

# =====================================
# PATHS
# =====================================

ARTIFACTS = Path("artifacts")

MODEL_PATH = ARTIFACTS / "models" / "kmeans_model.pkl"
SCALER_PATH = ARTIFACTS / "scalers" / "rfm_scaler.pkl"
SCALED_DATA_PATH = ARTIFACTS / "data" / "rfm_dataset.csv"
RAW_DATA_PATH = ARTIFACTS/ "data" / "rfm_raw.csv"
SUMMARY_PATH = ARTIFACTS / "cluster_profiles.csv"
METRICS_PATH = ARTIFACTS /"model_metrics.csv"
LOG_PATH =Path("logs") /"prediction_logs.csv"
CLUSTER_PATH = ARTIFACTS / "rfm_clustered.csv"
REPORTS = Path("reports")

# =====================================
# LOAD MODEL
# =====================================

def load_model():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler

# =====================================
# LOAD DATASET
# =====================================

def load_raw_dataset():
    return pd.read_csv(RAW_DATA_PATH)

def load_scaled_dataset():
    return pd.read_csv(SCALED_DATA_PATH)

# =====================================
# LOAD CLUSTER SUMMARY
# =====================================

def load_cluster_profiles():

    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)

    return pd.DataFrame()

# =====================================
# LOAD METRICS
# =====================================

def load_metrics():

    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)

    return pd.DataFrame()

def load_clustered_dataset():
    return pd.read_csv(CLUSTER_PATH)

# =====================================
# LOAD PREDICTION LOGS
# =====================================

def load_logs():

    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)

    return pd.DataFrame()

# =====================================
# PREDICT CUSTOMER
# =====================================

def predict_customer(recency, frequency, monetary):

    model, scaler = load_model()

    sample = pd.DataFrame({

        "Recency":[recency],
        "Frequency":[frequency],
        "Monetary":[monetary]

    })

    scaled = scaler.transform(sample)

    cluster = int(model.predict(scaled)[0])

    return cluster

# =====================================
# CUSTOMER SEGMENT
# =====================================

def segment_name(cluster):

    names = {

        0:"New Customers",

        1:"VIP Customers",

        2:"At Risk Customers",

        3:"Inactive Customers"

    }

    return names.get(cluster,"Unknown")

# =====================================
# API HEALTH
# =====================================

def api_health():

    try:

        response = requests.get(
            "http://127.0.0.1:8000/health",
            timeout=3
        )

        return response.status_code == 200

    except:

        return False

# =====================================
# LAST MODEL UPDATE
# =====================================

def model_last_updated():

    if MODEL_PATH.exists():

        return datetime.fromtimestamp(

            MODEL_PATH.stat().st_mtime

        )

    return None

# =====================================
# MODEL SIZE
# =====================================

def model_size():

    if MODEL_PATH.exists():

        return round(

            MODEL_PATH.stat().st_size/1024/1024,

            2

        )

    return 0

# =====================================
# LIST REPORTS
# =====================================

def available_reports():

    if REPORTS.exists():

        return os.listdir(REPORTS)

    return []

# =====================================
# LOAD JSON REPORT
# =====================================

def load_json(path):

    with open(path) as f:

        return json.load(f)

# =====================================
# TOTAL ARTIFACTS
# =====================================

def artifact_count():

    count=0

    for root,dirs,files in os.walk("artifacts"):

        count += len(files)

    return count

def table_style(df):
    """
    Apply a common dark theme to all dashboard tables.
    """

    return (
        df.style
        .set_properties(**{
            "background-color": "#0f172a",
            "color": "white",
            "border": "1px solid #1e293b",
            "text-align": "center",
            "font-size": "14px"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1e293b"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #334155"),
                    ("padding", "8px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "8px")
                ]
            },
            {
                "selector": "tr:nth-child(even)",
                "props": [
                    ("background-color", "#111827")
                ]
            },
            {
                "selector": "tr:hover",
                "props": [
                    ("background-color", "#1d4ed8")
                ]
            }
        ])
    )