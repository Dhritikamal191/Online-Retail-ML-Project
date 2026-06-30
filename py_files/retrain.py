import subprocess

from logger import logger

def retrain():
    logger.info("Starting model retraining...")

    subprocess.run(["python","train.py"], check=rue)

    logger.info("Retraining completed successfully.")

if __name__ == "__main__":
   retrain()
