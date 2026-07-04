import streamlit as st
import pandas as pd
from utils import (load_raw_dataset,load_scaled_dataset,load_metrics,
load_cluster_profiles)

def home_page():
    st.set_page_config(page_title="Online Retail ML Dashboard",page_icon="🛒",layout="wide")
    st.title("🛒 Online Retail Customer Segmentation")
    st.markdown("### End-to-End MLOps Dashboard")
    
    # -----------------------
    # Load Data
    # -----------------------
    
    raw_df = load_raw_dataset()
    scaled_df = load_scaled_dataset()
    metrics = load_metrics()
    profiles = load_cluster_profiles()
    
    # -----------------------
    # Sidebar
    # -----------------------
    
    st.sidebar.title("Dashboard")
    st.sidebar.success("Pipeline Status: ✅ Running")
    
    # -----------------------
    # KPIs
    # -----------------------

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers",len(scaled_df))
    col2.metric("Clusters",scaled_df["Cluster"].nunique())
    col3.metric("Average Recency",round(scaled_df["Recency"].mean(), 2))
    col4.metric("Average Monetary",round(scaled_df["Monetary"].mean(), 2))
    
    st.divider()

    # -----------------------
    # Dataset Preview
    # -----------------------

    st.subheader("Dataset Preview")
    st.dataframe(scal3d_df.head(10),use_container_width=True)

    st.divider()

    # -----------------------
    # Cluster Profiles
    # -----------------------

    st.subheader("Cluster Profiles")
    st.dataframe(profiles,use_container_width=True)

    st.divider()

    # -----------------------
    # Model Metrics
    # -----------------------

    st.subheader("Model Metrics")
    st.dataframe(metrics,use_container_width=True)

    st.divider()

    # -----------------------
    # Dataset Information
    # -----------------------

    left, right = st.columns(2)

    with left:
         st.info(f"""Total Customers : **{len(df)}**
         Total Features : **{scaled_df.shape[1]}**
         Clusters : **{scaled_df['Cluster'].nunique()}**
         """)

    with right:

         st.success("""
             ✅ Data Ingestion

             ✅ Data Validation

             ✅ Feature Engineering

             ✅ Model Training

             ✅ Drift Detection

             ✅ MLflow Tracking

             ✅ Automatic Retraining Ready
             """)

    st.divider()

    st.caption("Developed using Streamlit • MLflow • Evidently AI • Docker • GitHub Actions")
