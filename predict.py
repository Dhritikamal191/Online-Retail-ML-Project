import joblib
import pandas as pd
from datetime import datetime
model=joblib.load("artifacts/models/kmeans_model.pkl")
scaler=joblib.load("artifacts/scalers/rfm_scaler.pkl")

def predict_cluster(data):
    df=pd.DataFrame([{"Recency": recency, "Frequency": frequency, "Monetary": monetary, "AverageageOrderValue": average_order_value, "CustomerValue": customer_value}])
    print("Input columns:", df.columns.tolist())
    if hasattr(scaler, "feature_names_in_"):
       print("Expected columns:", scaler.feature_names_in_)
    scaled = scaler.transform(df)
    cluster = model.predict(scaled)
    return int(cluster[0])

if __name__ == "__main__":
   test = {"Recency": 30, "Frequency": 5, "Monetary": 500}

   print(predict_cluster(test))