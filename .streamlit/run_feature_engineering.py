import pandas as pd

from src.feature_engineering import FeatureEngineering
from src.config import load_config

config = load_config()

df = pd.read_csv(
    config["data"]["processed_data"]
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

feature = FeatureEngineering(df)

rfm, scaled = feature.run()

print(rfm.head())

print(scaled.head())