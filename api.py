from fastapi import FastAPI
from schemas import CustomerFeatures
from predict import predict_cluster
import os 
from datetime import datetime
import joblib
import pandas as pd

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

    cluster = predict_cluster(customer.dict())

    log = customer.dict()
    log["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%H:%S")
    log["PredictedCluster"] = int(cluster)
    log_df = pd.DataFrame([log])

    if os.path.exists(LOG_FILE):
       log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
         log_df.to_csv(LOG_FILE, index=False)

    return {"cluster": cluster}