import os 
import pandas as pd
from retrain import retrain

TRAIN_FILE = "artifacts/data/rfm_raw.csv"
LOG_FILE = "artifacts/logs/prediction_logs.csv"

def detect_drift():

    if not os.path.exists(TRAIN_FILE):
           return {"Error": "Training dataset not found."}

    if not os.path.exists(LOG_FILE):
           return {"Error": "Prediction logs not found."}

    train = pd.read_csv(TRAIN_FILE)
    prod =pd.read_csv(LOG_FILE)

    features = ["Recency", "Frequency", "Monetary", "AverageOrderValue", "CustomerValue"]

    reports = {}

    for feature in features:
        train_mean = train[feature].mean()
        prod_mean = prod[feature].mean()
        drift_percent = abs(prod_mean - train_mean) / train_mean * 100
        reports[feature] = {"Training Mean": round(train_mean,2), "Production Mean": round(prod_mean,2), "Drift (%)": round(drift_percent,2), "Status": "Drift Detected" if drift_percent > 20 else "Stable"}
 
        os.makedirs("artifacts/drift", exist_ok=True)
        drift_df = pd.DataFrame(reports).T
        drift_df.index.name ="Feature"
        drift_df.reset_index(inplace=True)
        drift_df.to_csv("artifacts/drift/drift_report.csv", index=False)
        print("Drift report saved to artifacts/drift/drift_reports.csv")
        print("Drift report saved.")

        drift_count = (drift_df["Status"] == "Drift").sum()

        if drift_count >= 3:
           print("Significant drift detected. Starting retraining...")
           retrain()
        else:
             print("Drift is within acceptable limits.")

    return drift_df.to_dict(orient="records")

if __name__ == "__main__":
    print(detect_drift())