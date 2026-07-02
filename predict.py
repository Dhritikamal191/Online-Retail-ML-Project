import joblib
import pandas as pd

model = joblib.load("artifacts/models/kmeans_model.pkl")
scaler = joblib.load("artifacts/scalers/rfm_scaler.pkl")

SEGMENT_NAMES = {
    0: "New Customers",
    1: "VIP Customers",
    2: "At Risk Customers",
    3: "Inactive Customers"
}

def predict_cluster(data):

    recency = data["Recency"]
    frequency = data["Frequency"]
    monetary = data["Monetary"]

    # Engineer the missing features
    average_order_value = monetary / frequency if frequency > 0 else 0
    customer_value = frequency * average_order_value

    df = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AverageOrderValue": average_order_value,
        "CustomerValue": customer_value
    }])

    scaled = scaler.transform(df)

    cluster = int(model.predict(scaled)[0])

    segment = SEGMENT_NAMES.get(cluster, "Unknown")

    distance = float(model.transform(scaled).min())

    return {
        "cluster": cluster,
        "segment": segment,
        "distance": distance
    }