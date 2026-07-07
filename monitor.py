import pandas as pd
import os

LOG_FILE = "artifacts/logs/prediction_logs.csv"

def get_monitoring_metrics():

    if not os.path.exists(LOG_FILE):
       return {"Total Predictions": 0, "Latest Prediction": None, "Cluster Distribution": {}}
 
    df = pd.read_csv(LOG_FILE, engine= "python",on_bad_lines="skip")

    metrics = {"Total Predictions": len(df), "Latest Prediction": df.iloc[-1].to_dict(), "Cluster Distribution": df["PredictedCluster"].value_counts().to_dict()}

    return metrics
  
print(get_monitoring_metrics())