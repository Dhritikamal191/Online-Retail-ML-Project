from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessor

ingestion = DataIngestion()

df = ingestion.load_data()

processor = DataPreprocessor(df)

clean_df = processor.preprocess()

print(clean_df.head())

print(clean_df.shape)