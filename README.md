## 🛍️ Online Retail Customer Segmentation using Machine Learning & MLOps

### 📌 Project Overview
This project develops an end-to-end Machine Learning Operations (MLOps) pipeline for customer segmentation using the Online Retail Dataset.

The system automatically preprocesses customer transaction data, performs RFM (Recency, Frequency, Monetary) analysis, predicts customer segments using K-Means Clustering, monitors prediction quality, detects data drift, and supports automated model retraining.
The project demonstrates a production-ready ML workflow integrating FastAPI, Streamlit, Docker, GitHub Actions, MLflow, and Evidently AI.

### 🚀 Live Demo
🌐 Streamlit Dashboard

[![Live Dashboard](https://img.shields.io/badge/🚀_Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge)](https://online-retail-ml-project-6amdxys7t5plfppj5bujck.streamlit.app/)

### ⚡ FastAPI Swagger

[![FastAPI Docs](https://img.shields.io/badge/📖_Swagger_API-FastAPI-009688?style=for-the-badge)](https://online-retail-ml-project-production.up.railway.app/docs#)

### 🎯 Objectives
Perform customer segmentation using RFM analysis
Build an automated ML pipeline
Deploy prediction API
Develop an interactive business dashboard
Monitor production predictions
Detect data drift
Support automated retraining
Demonstrate complete MLOps workflow

### 📂 Project Structure
Online-Retail-ML-Project
│
├── api.py
├── app.py
├── predict.py
├── train.py
├── retrain.py
├── monitor.py
├── drift_detection.py
├── schemas.py
├── config.yaml
│
├── dashboard/
│   ├── home.py
│   ├── prediction.py
│   ├── eda.py
│   ├── model_dashboard.py
│   ├── mlops.py
│   ├── reports.py
│   └── utils.py
│
├── artifacts/
│   ├── data/
│   ├── models/
│   ├── drift/
│   ├── logs/
│   └── metrics/
│
├── requirements.txt
└── README.md

### 📊 Dataset
Dataset: Online Retail Dataset
Features
InvoiceNo
CustomerID
InvoiceDate
Quantity
UnitPrice
Country

### 🧹 Feature Engineering
Customer behaviour is transformed into RFM features.
Recency
Frequency
Monetary
Average Order Value
Customer Value

### 🤖 Machine Learning Model
Algorithm:
K-Means Clustering
Segments include:
👑 VIP Customers
🆕 New Customers
⚠️ At Risk Customers
🔴 Inactive Customers
📈 Interactive Dashboard

The Streamlit dashboard includes:
- Customer Segmentation Prediction
- Customer Persona Analysis
- Business Recommendations
- RFM Radar Chart
- Prediction Confidence Gauge
- Prediction History
- Prediction Monitoring
- Data Drift Dashboard
- Model Monitoring
- Executive Reports
- Download Prediction Reports
- ⚙️ FastAPI
- REST API Endpoints
  - Endpoint Description
    /Home
    /health
    Health Check
    /predict
    Predict Customer Segment
    /monitor
    Model Monitoring
    /drift
    Data Drift Detection

### 📦 MLOps Pipeline
✔ Data Validation
✔ Data Preprocessing
✔ Feature Engineering
✔ Model Training
✔ Model Serialization
✔ Prediction Logging
✔ Monitoring
✔ Drift Detection
✔ Automatic Retraining
✔ API Deployment
✔ Interactive Dashboard

### 📊 Monitoring
Production monitoring includes
Prediction Count
Average Customer Value
Average Order Value
Cluster Distribution
Segment Distribution
Prediction Logs

### 🔍 Drift Detection
The application compares
Training Data
vs
Production Data
using
Recency
Frequency
Monetary
Average Order Value
Customer Value
Automatic retraining is triggered whenever significant drift is detected.

### 📈 Technologies Used
Programming
Python
Machine Learning
Scikit-learn
Data Processing
Pandas
NumPy
Visualization
Plotly
Streamlit
API
FastAPI
MLOps
MLflow
Docker
GitHub Actions
Evidently AI

### 📊 Deployment
Streamlit
FastAPI
Railway / Render
GitHub

### 💼 Business Value
This solution enables businesses to:
Identify high-value customers
Improve customer retention
Detect customer behavior changes
Automate segmentation
Monitor production model performance
Detect data drift
Retrain models automatically
Improve marketing efficiency

### 👨‍💻 Author
Dhritikamal Das
M.Sc. MACS
Machine Learning Enthusiast | MLOps Enthusiast
GitHub: [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Dhritikamal191)

LinkedIn: [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dhritikamal-das-9912652a5)

### ⭐ Future Enhancements
Deep Learning-based Customer Segmentation
Real-time Streaming Predictions
Cloud Storage Integration
Prometheus Monitoring
Grafana Dashboards
Kubernetes Deployment
Automated CI/CD Pipeline Enhancements
Email Alerts for Model Drift
Multi-model Experiment Tracking

### 📜 License
This project is intended for educational and portfolio purposes.
