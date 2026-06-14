import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ==========================================================
# LOAD ARTIFACTS
# ==========================================================

@st.cache_data
def load_data():
    rfm = pd.read_csv("rfm_dataset.csv")
    comparison = pd.read_csv("algorithm_comparison.csv")
    profiles = pd.read_csv("cluster_profiles.csv")
    pca_df = pd.read_csv("pca_data.csv")

    return rfm, comparison, profiles, pca_df


@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    pca = joblib.load("pca.pkl")

    return scaler, kmeans, pca


rfm, comparison, profiles, pca_df = load_data()
scaler, kmeans, pca = load_models()

with open("best_k.txt", "r") as f:
    best_k = f.read()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🛍️ Customer Segmentation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset Overview",
        "RFM Analysis",
        "Elbow & Silhouette",
        "Algorithm Comparison",
        "Cluster Visualization",
        "Cluster Profiles",
        "Segment Predictor",
        "Business Recommendations"
    ]
)

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

if page == "Project Overview":

    st.title("🛍️ Customer Segmentation Dashboard")

    st.markdown("""
    ### Project Objective

    Segment customers based on purchasing behaviour using
    **RFM Analysis** and **Clustering Techniques**.

    This project evaluates multiple clustering algorithms:

    - ✅ K-Means
    - ✅ Hierarchical Clustering
    - ✅ DBSCAN

    and selects the optimal model for deployment.

    ### Methodology

    1. Data Cleaning
    2. RFM Feature Engineering
    3. Log Transformation
    4. Standardization
    5. Optimal K Selection
    6. Clustering
    7. Model Evaluation
    8. Business Recommendations
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        f"{rfm.shape[0]:,}"
    )

    col2.metric(
        "Optimal K",
        best_k
    )

    col3.metric(
        "Algorithms Compared",
        3
    )

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{rfm.shape[0]:,}"
    )

    col2.metric(
        "Features",
        4
    )

    col3.metric(
        "Clusters",
        len(rfm["Cluster"].unique())
    )

    st.subheader("RFM Dataset")

    st.dataframe(
        rfm.head(20),
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        rfm.describe(),
        use_container_width=True
    )

# ==========================================================
# RFM ANALYSIS
# ==========================================================

elif page == "RFM Analysis":

    st.title("📈 RFM Analysis")

    tab1, tab2, tab3 = st.tabs(
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    )

    with tab1:

        st.subheader("Recency Distribution")

        fig = px.histogram(
            rfm,
            x="Recency",
            nbins=30,
            title="Recency Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Lower Recency indicates more recent purchases."
        )

    with tab2:

        st.subheader("Frequency Distribution")

        fig = px.histogram(
            rfm,
            x="Frequency",
            nbins=30,
            title="Frequency Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Higher Frequency indicates repeat customers."
        )

    with tab3:

        st.subheader("Monetary Distribution")

        fig = px.histogram(
            rfm,
            x="Monetary",
            nbins=30,
            title="Monetary Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Higher Monetary value indicates high-spending customers."
        )