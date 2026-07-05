from fastapi import FastAPI
from schemas import CustomerFeatures
from predict import predict_cluster
import os 
from datetime import datetime
import joblib
import pandas as pd
from monitor import get_monitoring_metrics
from drift_detection import detect_drift

app = FastAPI(title="Online Retail Customer Segmentation API", version="1.0")

model = joblib.load("artifacts/models/kmeans_model.pkl")

LOG_DIR = "artifacts/logs"
LOG_FILE = os.path.join(LOG_DIR,"prediction_logs.csv")
os.makedirs(LOG_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Online Retail API Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer:
CustomerFeatures):

    result = predict_cluster(customer.dict())

    cluster = result["cluster"]
    
    segment = rseult["segment"]

    log = customer.dict()

    log["AverageOrderValue"] = (
    log["Monetary"] / log["Frequency"]
    if log["Frequency"] > 0 else 0)

    log["CustomerValue"] = (
    log["Frequency"] * log["AverageOrderValue"])

    log["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log["PredictedCluster"] = cluster
    log["Segment"] = segment
    log_df = pd.DataFrame([log])

    if os.path.exists(LOG_FILE):
       log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
         log_df.to_csv(LOG_FILE, index=False)

    return {"cluster": cluster,"segment": segment}

@app.get("/monitor")
def monitor():
    return get_monitoring_metrics()

@app.get("/drift")
def drift():
    return detect_drift()