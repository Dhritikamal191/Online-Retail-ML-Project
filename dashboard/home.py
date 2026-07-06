import streamlit as st
import pandas as pd
from utils import (
    load_clustered_dataset,
    load_metrics,
    load_cluster_profiles,
    model_last_updated,
    model_size,
    artifact_count,
    show_table 
)

def home_page():

    st.title("🛍️ Online Retail Customer Segmentation Platform")
    st.markdown("### End-to-End Machine Learning & MLOps Dashboard")

    # -----------------------------
    # Load Data
    # -----------------------------
    df = load_clustered_dataset()
    metrics = load_metrics()
    profiles = load_cluster_profiles()

    # -----------------------------
    # Sidebar
    # -----------------------------
    st.sidebar.success("✅ Pipeline Running")

    # -----------------------------
    # KPI Cards
    # -----------------------------
        
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Customers", len(df))
    col2.metric("Clusters", df["Cluster"].nunique())
    col3.metric("Avg Recency", round(df["Recency"].mean(), 2))
    col4.metric("Avg Frequency", round(df["Frequency"].mean(), 2))
    col5.metric("Avg Monetary", f"${df['Monetary'].mean():,.2f}")

    st.divider()

    # -----------------------------
    # Dataset Preview
    # -----------------------------
    st.subheader("Dataset Preview")
    
    show_table(df.head(10))

    st.divider()

    # -----------------------------
    # Cluster Summary
    # -----------------------------
    st.subheader("Cluster Profiles")

    if not profiles.empty:
       show_table(profiles)
    else:
        st.warning("Cluster profile file not found.")

    st.divider()

    # -----------------------------
    # Model Performance
    # -----------------------------
    st.subheader("Model Comparison")

    if not metrics.empty:
        show_table(metrics)
    else:
        st.warning("Model metrics not found.")

    st.divider()

    # -----------------------------
    # Project Information
    # -----------------------------
    left, right = st.columns(2)

    with left:

        st.markdown("### Dataset Summary")

        st.write(f"**Total Customers:** {len(df)}")
        st.write(f"**Features:** {df.shape[1]}")
        st.write(f"**Clusters:** {df['Cluster'].nunique()}")
        if "CustomerValue" in df.columns:
           customer_value = df["CustomerValue"].mean()
        else:
             customer_value = (df["Frequency"] * df["Monetary"]).mean()

        st.write(f"**Average Customer Value:** ${customer_value:,.2f}")
    with right:

        st.markdown("### Model Information")

        st.write(f"**Last Updated:** {model_last_updated()}")
        st.write(f"**Model Size:** {model_size()} MB")
        st.write(f"**Artifacts:** {artifact_count()}")

    st.divider()

    # -----------------------------
    # Pipeline Status
    # -----------------------------
    st.subheader("Pipeline Status")

    st.success("""
    ✅ Data Ingestion

    ✅ Data Validation

    ✅ Feature Engineering

    ✅ Customer Segmentation

    ✅ Model Evaluation

    ✅ MLflow Experiment Tracking

    ✅ Drift Detection

    ✅ Automatic Retraining

    ✅ FastAPI Deployment

    ✅ CI/CD with GitHub Actions
    """)

    st.divider()

    st.caption(
        "Built with Streamlit • Scikit-learn • MLflow • FastAPI • Docker • GitHub Actions"
    )