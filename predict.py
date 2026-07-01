import joblib
import pandas as pd
from datetime import datetime
model=joblib.load("artifacts/models/kmeans_model.pkl")
scaler=joblib.load("artifacts/scalers/rfm_scaler.pkl")

def predict_cluster(data):
    df=pd.DataFrame([data])
    scaled = scaler.transform(df)
    cluster = model.predict(scaled)
    return int(cluster[0])