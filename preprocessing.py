import os
import pandas as pd
from logger import logger
from config import load_config

config = load_config()

class DataPreprocessor:

    def __init__(self, df):

        self.df = df.copy()

    def remove_missing_customer(self):

        logger.info("Removing Missing CustomerID")

        before = len(self.df)

        self.df = self.df.dropna(subset=["CustomerID"])

        after = len(self.df)

        logger.info(f"Removed {before-after} rows")

    def remove_cancelled_orders(self):

        logger.info("Removing Cancelled Orders")

        before = len(self.df)

        self.df = self.df[
            ~self.df["InvoiceNo"].astype(str).str.startswith("C")
        ]

        after = len(self.df)

        logger.info(f"Removed {before-after} cancelled invoices")

    def remove_invalid_quantity(self):

        logger.info("Removing Invalid Quantity")

        before = len(self.df)

        self.df = self.df[self.df["Quantity"] > 0]

        after = len(self.df)

        logger.info(f"Removed {before-after} rows")

    def remove_invalid_price(self):

        logger.info("Removing Invalid UnitPrice")

        before = len(self.df)

        self.df = self.df[self.df["UnitPrice"] > 0]

        after = len(self.df)

        logger.info(f"Removed {before-after} rows")

    def convert_date(self):

        logger.info("Converting InvoiceDate")

        self.df["InvoiceDate"] = pd.to_datetime(
            self.df["InvoiceDate"]
        )

    def create_total_price(self):

        logger.info("Creating TotalPrice Feature")

        self.df["TotalPrice"] = (
            self.df["Quantity"] *
            self.df["UnitPrice"]
        )

    def save(self):

        path = config["data"]["processed_data"]

        directory=os.path.dirname(path)
  
        if directory:
           os.makedirs(directory, exist_ok=True)

        self.df.to_csv(path, index=False)

        logger.info(f"Processed Dataset Saved : {path}")

    def preprocess(self):

        self.remove_missing_customer()

        self.remove_cancelled_orders()

        self.remove_invalid_quantity()

        self.remove_invalid_price()

        self.convert_date()

        self.create_total_price()

        self.save()

        logger.info("Preprocessing Completed Successfully")

        return self.df

if __name__ == "__main__":
   from data_ingestion import DataIngestion
   print("Main block running")
   ingestion = DataIngestion()
   df = ingestion.load_data()
   preprocessor = DataPreprocessor(df)
   clean_df = preprocessor.preprocess()
   print(clean_df.head)
   print(clean_df.shape)