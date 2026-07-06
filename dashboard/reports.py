import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from utils import show_table

def reports_page():

    st.title("📁 Reports & Downloads")

    st.markdown("View and download project reports, logs and artifacts.")

    st.divider()

    ######################################
    # Prediction Logs
    ######################################

    st.header("📜 Prediction Logs")

    log_path = "artifacts/logs/prediction_logs.csv"

    if os.path.exists(log_path):
       logs = pd.read_csv(log_path)

       show_table(logs)

       csv = logs.to_csv(index=False).encode()

       st.download_button(
        "⬇ Download Prediction Logs",
        csv,
        "prediction_logs.csv",
        "text/csv"
        )
    else:
         st.info("Prediction logs not available.")
      
    #######################################
    # Model Metrics
    #######################################

    st.divider()

    st.header("📊 Model Metrics")

    metrics_path = "artifacts/model_metrics.csv"

    if os.path.exists(metrics_path):

       metrics = pd.read_csv(metrics_path)

       show_table(metrics)

    else:

         st.warning("Metrics not found.")

    #########################################
    # Cluster Profiles
    #########################################

    st.divider()

    st.header("👥 Cluster Profiles")

    profile_path = "artifacts/cluster_profiles.csv"

    if os.path.exists(profile_path):

       profile = pd.read_csv(profile_path)

       show_table(profile)

    else:

         st.warning("Cluster profile not found.")

    #######################################
    # Drift Report
    #######################################

    st.divider()

    st.header("📈 Drift Report")

    html_report = "artifacts/drift/drift_report.csv"

    if os.path.exists(html_report):

       with open(html_report, "r", encoding="utf-8") as f:

            html = f.read()

            st.components.v1.html(html,height =700,scrolling=True)

    else:

         st.info("Drift report unavailable.")

    ######################################
    # JSON Report
    ######################################

    json_report = "reports/drift_metrics.json"

    if os.path.exists(json_report):

       with open(json_report) as f:

            st.json(json.load(f))

    ########################################
    # Artifact Explorer
    ########################################

    st.divider()

    st.header("📂 Artifact Explorer")

    files = []

    for root, dirs, fs in os.walk("artifacts"):

        for file in fs:

            path = os.path.join(root, file)

            files.append({"File": file,"Folder": root,"Size (KB)": round(os.path.getsize(path)/1024,2)})

    if len(files):

        show_table(pd.DataFrame(files))

    ########################################
    # Download Center
    ########################################

    st.divider()

    st.header("⬇ Download Center")

    download_files = [

    "artifacts/model_metrics.csv",

    "artifacts/cluster_profiles.csv",

    "artifacts/rfm_clustered.csv"

    ]

    for file in download_files:

        if os.path.exists(file):

           with open(file, "rb") as f:

                st.download_button(f"Download {Path(file).name}",f,Path(file).name)

    st.divider()

    st.success("Reports Dashboard Loaded Successfully")
