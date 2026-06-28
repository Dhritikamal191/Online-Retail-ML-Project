import os
import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from src.logger import logger
from src.config import load_config

config = load_config()


class TrainModels:

    def __init__(self):

        self.df = pd.read_csv(
            "artifacts/rfm_features.csv"
        )

        self.X = self.df.drop(
            columns=["CustomerID"]
        )

    def evaluate(self, name, labels):

        # Skip invalid clustering results
        if len(set(labels)) < 2:
            logger.warning(f"{name} produced only one cluster.")
            return None

        score = {

            "Model": name,

            "Silhouette":
                silhouette_score(self.X, labels),

            "Davies":
                davies_bouldin_score(self.X, labels),

            "Calinski":
                calinski_harabasz_score(self.X, labels)

        }

        return score

    def train(self):

        models = {

            "KMeans": KMeans(

                n_clusters=config["model"]["kmeans"]["n_clusters"],

                random_state=config["model"]["random_state"]

            ),

            "Agglomerative":

                AgglomerativeClustering(

                    n_clusters=config["model"]["agglomerative"]["n_clusters"]

                ),

            "DBSCAN":

                DBSCAN(

                    eps=config["model"]["dbscan"]["eps"],

                    min_samples=config["model"]["dbscan"]["min_samples"]

                )

        }

        results = []

        best_model = None

        best_labels = None

        best_score = -999

        for name, model in models.items():

            logger.info(f"Training {name}")

            labels = model.fit_predict(self.X)

            metrics = self.evaluate(name, labels)

            if metrics is None:
                continue

            results.append(metrics)

            if metrics["Silhouette"] > best_score:

                best_score = metrics["Silhouette"]

                best_model = model

                best_labels = labels

        os.makedirs(
            "artifacts/models",
            exist_ok=True
        )

        joblib.dump(

            best_model,

            "artifacts/models/best_model.pkl"

        )

        self.df["Cluster"] = best_labels

        self.df.to_csv(

            "artifacts/rfm_clustered.csv",

            index=False

        )

        pd.DataFrame(results).to_csv(

            "artifacts/model_metrics.csv",

            index=False

        )

        logger.info("Best Model Saved")

        return pd.DataFrame(results)