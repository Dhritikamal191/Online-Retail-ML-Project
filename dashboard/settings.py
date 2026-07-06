import streamlit as st
import platform
import sys
import sklearn
import pandas
import joblib
import plotly
import json

def settings_page():

    st.title("⚙ Settings")

    st.markdown("Application configuration and project information.")

    st.divider()

    # ================================
    # PROJECT
    # ================================

    st.header("📌 Project")

    st.info("""

    Project Name

    Online Retail Customer Segmentation

    Type

    End-to-End Machine Learning + MLOps

    Developer

    Dhritikamal Das

    """)

    # =============================== 
    # TECHNOLOGIES
    # ===============================

    st.divider()

    st.header("🛠 Technologies")

    tech = [

    "Python",

    "Scikit-learn",

    "Pandas",

    "Plotly",

    "Streamlit",

    "FastAPI",

    "MLflow",

    "Evidently AI",

    "Docker",

    "GitHub Actions"

    ]

    for t in tech:

        st.markdown(f"✅ {t}")

    # ==================================
    # PACKAGE VERSIONS
    # ==================================

    st.divider()

    st.header("📦 Installed Packages")

    versions = {

    "Python": sys.version.split()[0],

    "Pandas": pandas.__version__,

    "Scikit-learn": sklearn.__version__,

    "Plotly": plotly.__version__,

    "Joblib": joblib.__version__

    }

    st.code(
    json.dumps(versions, indent=4),
    language="json")

    # =====================================
    # SYSTEM
    # =====================================

    st.divider()

    st.header("💻 System Information")

    system = {

    "Platform": platform.system(),

    "Release": platform.release(),

    "Machine": platform.machine(),

    "Processor": platform.processor()

    }

    st.code(
    json.dumps(system, indent=4),
    language="json")

    # =================================
    # PIPELINE
    # =================================

    st.divider()

    st.header("🔄 Pipeline")
  
    st.success("""

    ✔ Data Ingestion

    ✔ Data Validation

    ✔ Preprocessing

    ✔ Feature Engineering

    ✔ KMeans Training

    ✔ Prediction

    ✔ Monitoring

    ✔ Drift Detection

    ✔ Automatic Retraining

    ✔ MLflow

    ✔ Docker
 
    ✔ FastAPI
  
    ✔ GitHub Actions

    """)

    # ==================================
    # FUTURE
    # ==================================

    st.divider()

    st.header("🚀 Future Improvements")

    future = [

    "Model Registry",

    "Kubernetes Deployment",

    "Cloud Deployment",

    "Authentication",

    "Role Based Access",

    "Email Alerts",

    "Grafana Dashboard"

    ]

    for item in future:

        st.markdown(f"🔹 {item}")

    # ================================
    # FOOTER
    # ================================

    st.divider()

    st.caption("Retail Customer Segmentation | Version 1.0")
