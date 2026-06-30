from data_ingestion import DataIngestion
from data_validation import DataValidation

ingestion = DataIngestion()

df = ingestion.load_data()

validator = DataValidation(df)

validator.validate()

print("Validation Completed Successfully")