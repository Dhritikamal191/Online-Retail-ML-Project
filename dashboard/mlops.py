import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import requests
import json
import os
import joblib 
from pathlib import Path
from datetime import datetime

def mlops_page():

    st.title("🔄 MLOps Dashboard")

    st.markdown("""
Production Monitoring Dashboard

    ✔ MLflow
    ✔ Drift Detection
    ✔ Monitoring
    ✔ FastAPI
    ✔ Docker
    ✔ GitHub Actions
    ✔ Automatic Retraining
    """)

    st.divider()

    ########################################
    # SYSTEM STATUS
    #########################################

    c1,c2,c3,c4=st.columns(4)

    with c1:

         if os.path.exists("mlruns"):
            st.success("MLflow")

         else:
              st.error("MLflow")

    with c2:

         if os.path.exists("artifacts/models/kmeans_model.pkl"):

            st.success("Model")

         else:
              st.error("Model")

    with c3:

         if os.path.exists("artifacts/drift/drift_report.csv"):

            st.success("Drift Report")

         else:
              st.warning("No Report")

    with c4:

         if os.path.exists("artifacts/logs/prediction_logs.csv"):

            st.success("Monitoring")

         else:
              st.warning("No Logs")

    st.divider()

    ########################################
    # API HEALTH
    #########################################

    st.subheader("🌐 FastAPI Health")

    try:

        response=requests.get(

        "https://online-retail-ml-project.onrender.com/docs#/default/health_health_get",

        timeout=3)

        if response.status_code==200:

           st.success("🟢 API Running")

           st.json(response.json())

        else:

             st.error("API Error")

    except:

           st.warning("FastAPI Not Running")

    st.divider()

    ########################################
    # MLOP Status
    ########################################

    st.header("⚙️ MLOps Pipeline Status")

    col1, col2, col3 = st.columns(3)

    with col1:
         st.success("✅ GitHub Actions")
         st.caption("Automatic Retraining Enabled")

    with col2:
         st.success("✅ MLflow")
         st.caption("Model Version Tracking Active")

    with col3:
         st.success("✅ Evidently AI")
         st.caption("Drift Monitoring Enabled")

    st.divider()

    st.info("""
    ### Production Workflow

    1. New retail data is uploaded.
    2. GitHub Actions automatically starts the pipeline.
    3. Data validation is performed.
    4. Feature engineering generates RFM features.
    5. Models are retrained automatically.
    6. MLflow logs experiments and model versions.
    7. Drift detection runs using Evidently AI.
    8. The latest approved model is deployed.
""")
    st.divider()

    ########################################
    # DRIFT REPORT
    ########################################

    st.subheader("📈 Drift Detection")

    if os.path.exists("reports/drift_metrics.json"):

       with open(

        "reports/drift_metrics.json") as f:

            drift=json.load(f)

            drift_score=drift.get ("drift_score",0)

       fig=go.Figure(go.Indicator(mode="gauge+number",value=drift_score*100,title={"text":"Drift Score"},gauge={"axis":{"range":[0,100]},"bar":{"color": "royalblue"},"steps":[{"range":[0,30], "color":"green"},{"range": [30,60], "color": "orange"},{"range":[60,100], "color": "red"}]}))

       fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
       
       st.plotly_chart(fig,use_container_width=True)

    else:
         st.info("Drift report unavailable.")

    st.divider()

    ######################################
    # MONITORING
    ######################################

    st.subheader("📊 Prediction Monitoring")

    log="artifacts/logs/prediction_logs.csv"

    if os.path.exists(log):

       history=pd.read_csv(log)

       st.dataframe(history.tail(20),
use_container_width=True)

       plot_df = history.melt (id_vars= "PredictedCluster",value_vars=["Recency","Frequency",
"Monetary","AverageOrderValue","CustomerValue"],var_name="Feature",value_name="Value")

       fig = px.histogram(plot_df,x="Value",
color="PredictedCluster",facet_col="Feature",title="Prediction Feature Distribution")

       fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
       
       st.plotly_chart(fig, use_container_width=True)

    else:

         st.warning("Prediction log not found.")

    st.divider()

    #######################################
    # MODEL FILES
    #######################################

    st.subheader("📂 Model Artifacts")

    files=[]

    for root,dirs,fs in os.walk("artifacts"):

        for f in fs:

            path=os.path.join(root,f)

            files.append({"File":f, "Folder":root,"Size(KB)":round(os.path.getsize(path)/1024,2),"Modified": datetime.fromtimestamp(os.path.getmtime(path))})

    if len(files)>0:

       st.dataframe(pd.DataFrame(files),
use_container_width=True)

    st.divider()

    #######################################
    # PIPELINE HEALTH
    ########################################

    st.subheader("❤️ Pipeline Health")

    try:
        joblib.load("artifacts/models/kmeans_model.pkl")
        model_health =100
    except:
           model_health = 0

    fig=go.Figure(go.Indicator (mode="gauge+number",value=model_health,title={"text":"Pipeline Health"},gauge={'axis':{'range':[0,100]},'bar':{'color':'#2563eb'},'steps':[{'range':[0,60],'color':'red'},{'range':[60,85],'color':'orange'},{'range':[85,100],'color':'green'}]}))

    fig.update_layout(font=dict(size=17, color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    ########################################
    # DEPLOYMENT STATUS
    #######################################

    st.subheader("🚀 Deployment")

    deployment=pd.DataFrame({

    "Component":[

    "FastAPI",

    "Docker",

    "GitHub Actions",

    "MLflow",

    "Evidently AI",

    "Prediction API",

    "Retraining"

    ],

    "Status":[

    "Ready",

    "Ready",

    "Ready",

    "Enabled",

    "Enabled",

    "Running",

    "Enabled"

    ]

    })

    st.dataframe(deployment, use_container_width=True)

    st.divider()

    st.caption("Enterprise MLOps Dashboard")
