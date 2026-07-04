import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logger import logger
from config import load_config

config = load_config()

class FeatureEngineering:

    def __init__(self, df):

        self.df = df.copy()

    def create_rfm(self):

        logger.info("Creating RFM Features")

        reference_date = pd.to_datetime(config["features"]["reference_date"])

        rfm = self.df.groupby("CustomerID").agg({"InvoiceDate":lambda x: (reference_date - x.max()).days,"InvoiceNo":"nunique","TotalPrice":"sum"})

        rfm.columns = ["Recency","Frequency","Monetary"]

        self.rfm = rfm.reset_index()

        logger.info("RFM Created")
        

    def additional_features(self):

        logger.info("Creating Business Features")

        self.rfm["AverageOrderValue"] = (self.rfm["Monetary"] / self.rfm["Frequency"])

        self.rfm["CustomerValue"] = (self.rfm["Frequency"] * self.rfm["AverageOrderValue"])

        self.rfm.to_csv("artifacts/data/rfm_raw.csv", index=False)
       
        logger.info("Raw RFM dataset saved")
    def scale_features(self):

        logger.info("Scaling Features")

        features = ["Recency","Frequency","Monetary","AverageOrderValue","CustomerValue"]

        scaler = StandardScaler()

        scaled = scaler.fit_transform(self.rfm[features])

        self.scaled_df = pd.DataFrame(scaled,columns=features)

        self.scaled_df["CustomerID"] = self.rfm["CustomerID"]

        os.makedirs("artifacts/scalers",exist_ok=True)

        joblib.dump(scaler,config["artifacts"]["scaler"])

        logger.info("Scaler Saved")

    def save_dataset(self):

        os.makedirs("artifacts",exist_ok=True)

        self.scaled_df.to_csv(config["artifacts"]["rfm_dataset"],index=False)

        logger.info("Feature Dataset Saved")

    def run(self):

        self.create_rfm()

        self.additional_features()

        self.scale_features()

        self.save_dataset()

        return self.rfm, self.scaled_df

config = load_config()
if __name__ == "__main__":
   from preprocessing import DataPreprocessor
   df = pd.read_csv(config["data"]["processed_data"])
   df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
   feature = FeatureEngineering(df)
   rfm, scaled = feature.run()
   print(rfm.head())
   print(scaled.head())