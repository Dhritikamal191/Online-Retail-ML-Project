import os

from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation

ingestion = DataIngestion()

df = ingestion.load_data()

validator = DataValidation(df)

validator.validate()

os.makedirs("reports", exist_ok=True)

missing.to_csv("reports/missing_values.csv")

duplicate_rows = self.df[self.df.duplicated()]

duplicate_rows.to_csv(
    "reports/duplicate_rows.csv",
    index=False
)

print("Validation Completed Successfully")