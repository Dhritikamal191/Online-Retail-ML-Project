import pandas as pd
from logger import logger
from config import load_config
import os
config = load_config()

class DataValidation:

    def __init__(self, df):

        self.df = df

        self.required_columns = config["validation"]["required_columns"]

    def validate_columns(self):

        logger.info("Checking Required Columns")

        missing = []

        for col in self.required_columns:

            if col not in self.df.columns:

                missing.append(col)

        if missing:

            raise Exception(f"Missing Columns : {missing}")

        logger.info("Column Validation Passed")

    def check_missing_values(self):

        logger.info("Checking Missing Values")

        missing = self.df.isnull().sum()

        logger.info(f"\n{missing}")

        os.makedirs("reports", exist_ok=True)

        missing.to_csv("reports/missing_values.csv")

        return missing

    def check_duplicates(self):

        logger.info("Checking Duplicate Rows")

        duplicates = self.df[self.df.duplicated()]

        os.makedirs("reports", exist_ok=True)

        duplicates.to_csv("reports/duplicate_rows.csv",index=False)
    
        duplicates = duplicates.shape[0]

        logger.info(f"Duplicate Rows : {duplicates}")

        return duplicates

    def check_negative_values(self):

        logger.info("Checking Negative Quantity and UnitPrice")

        negative_quantity = (self.df["Quantity"] <= 0).sum()

        negative_price = (self.df["UnitPrice"] <= 0).sum()

        logger.info(f"Negative Quantity : {negative_quantity}")

        logger.info(f"Negative Price : {negative_price}")

    def validate(self):

        self.validate_columns()

        self.check_missing_values()

        self.check_duplicates()

        self.check_negative_values()

        logger.info("Validation Completed Successfully")

if __name__ == "__main__":
   from data_ingestion import DataIngestion
   ingestion = DataIngestion()
   print("Main block running")
   df= ingestion.load_data()
   validator = DataValidation(df)
   validator.validate()
   print("Validation Completed")