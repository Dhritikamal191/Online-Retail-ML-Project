import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (load_metrics,model_last_updated,model_size,artifact_count, show_table)

def model_dashboard_page():

    st.title("🤖 Model Dashboard")

    st.markdown("""
    Monitor model performance, experiment tracking, training metrics and deployment status.
    """)

    st.divider()

    # ===================================
    # LOAD METRICS
    # ===================================

    metrics = load_metrics()

    # ====================================
    # KPI CARDS
    # ====================================

    c1,c2,c3,c4 = st.columns(4)

    with c1:
         st.metric("Model","KMeans")

    with c2:
         st.metric("Model Size", f"{model_size()} MB")

    with c3:
         last = model_last_updated()

         if last:
            st.metric("Last Training",
            last.strftime("%d-%m-%Y"))

    with c4:
         st.metric("Artifacts", artifact_count())

    st.divider()

    # =====================================
    # MODEL METRICS TABLE
    # =====================================

    st.subheader("Training Metrics")

    if not metrics.empty:
       show_table(metrics)

    else:
         st.warning("Metrics file not found.")

    st.divider()

    # =====================================
    # SILHOUETTE SCORE
    # =====================================

    if not metrics.empty:

       if "Silhouette" in metrics.columns:

           fig = px.bar(metrics,x="Model",y="Silhouette",color="Model",text="Silhouette",title="Silhouette Score")

           fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
           
           st.plotly_chart(fig,use_container_width=True)

    # =====================================
    # DAVIES BOULDIN
    # =====================================

    if not metrics.empty:

       if "Davies-Bouldin" in metrics.columns:

          fig = px.bar(metrics,x="Model",y="Davies-Bouldin",color="Model",title="Davies-Bouldin Index")

          fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          
          st.plotly_chart(fig,use_container_width=True)

    # ====================================
    # CALINSKI
    # =====================================

    if not metrics.empty:

       if "Calinski-Harabasz" in metrics.columns:

          fig = px.bar(metrics,x="Model",y="Calinski-Harabasz",color="Model",title="Calinski-Harabasz Score")

          fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          
          st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===================================
    # MODEL HEALTH
    # ===================================

    st.subheader("Model Health")

    try:
        joblib.load("artifacts/models/kmeans_model.pkl")
        model_health =100
    except:
           model_health =0

    fig = go.Figure(go.Indicator (mode="gauge+number",value=model_health,title={"text":"Overall Model Health"},gauge={'axis':{'range':[0,100]},'bar':{'color':"#2563eb"},'steps':[{'range':[0,60],'color':'red'},{'range':[60,85],'color':'orange'},{'range':[85,100],'color':'green'}]}))

    fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ======================================
    # MODEL INFORMATION
    # ======================================

    left,right = st.columns(2)

    with left:

         st.success("""

         Algorithm

         ✔ KMeans

         ✔ Unsupervised Learning

         ✔ RFM Clustering

         """)

    with right:

         st.success("""

         Deployment

         ✔ FastAPI

         ✔ Docker

         ✔ GitHub Actions

         ✔ MLflow

         """)

    st.divider()

    # =====================================
    # ARTIFACTS
    # =====================================

    st.subheader("Artifacts")

    artifact_data = pd.DataFrame({

    "Artifact":[

    "Model",

    "Scaler",

    "Metrics",

    "Cluster Profiles",

    "RFM Dataset"

    ],

    "Status":[

    "Available",

    "Available",

    "Available",

    "Available",

    "Available"

    ]
  
    })
  
    show_table(artifact_data)

    st.divider()

    # ======================================
    # MLFLOW STATUS
    # ======================================

    st.subheader("MLflow")

    if os.path.exists("mlruns"):

       st.success("✔ MLflow Experiments Available")

    else:

         st.warning("MLflow experiments not found.")

    if os.path.exists("mlflow.db"):

       st.success("✔ MLflow Database Connected")

    else:

         st.warning("MLflow Database Missing")

    st.divider()

    st.caption("Retail Customer Segmentation | Model Dashboard")
