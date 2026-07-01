import pandas as pd
from logger import logger
from config import load_config

config = load_config()

class DataIngestion:

    def __init__(self):
        self.path = config["data"]["raw_data"]

    def load_data(self):

        logger.info("Loading Dataset")

        df = pd.read_excel(self.path)

        logger.info(f"Dataset Loaded Successfully : {df.shape}")

        return df


if __name__ == "__main__":

    ingestion = DataIngestion()

    df = ingestion.load_data()

    print(df.head())