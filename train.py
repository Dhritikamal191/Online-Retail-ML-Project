import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import (silhouette_score,davies_bouldin_score,calinski_harabasz_score)
from logger import logger
from config import load_config

config = load_config()

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(config["mlflow"]["experiment_name"])

class TrainModels:

    def __init__(self):

        self.df = pd.read_csv("artifacts/data/rfm_raw.csv")

        self.X = self.df.drop(columns=["CustomerID"])

    def evaluate(self, name, labels):

        if len(set(labels)) < 2:
            logger.warning(f"{name} produced only one cluster.")
            return None

        score = {"Model": name,"Silhouette":silhouette_score(self.X, labels),"Davies":davies_bouldin_score(self.X, labels),"Calinski":calinski_harabasz_score(self.X, labels)}

        return score

    def train(self):

        models = {"KMeans": KMeans(n_clusters=config["model"]["kmeans"]["n_clusters"],random_state=config["model"]["random_state"]),"Agglomerative":AgglomerativeClustering(n_clusters=config["model"]["agglomerative"]["n_clusters"]),"DBSCAN":DBSCAN(eps=config["model"]["dbscan"]["eps"],min_samples=config["model"]["dbscan"]["min_samples"])}

        results = []

        best_model = None

        best_labels = None

        best_score = -999

        for name, model in models.items():

            with mlflow.start_run(run_name=name):

                logger.info(f"Training {name}")

                labels = model.fit_predict(self.X)

                if name == "KMeans":
                   joblib.dump(model,"artifacts/models/kmeans_model.pkl")
                   logger.info("KMeans model saved")

                metrics = self.evaluate(name, labels)

                if metrics is None:
                   continue

                results.append(metrics)

                mlflow.log_param("Model", name)

                if name == "KMeans":
                   mlflow.log_param("Clusters",config["model"]["kmeans"]["n_clusters"])

                elif name == "Agglomerative":
                     mlflow.log_param("Clusters",config["model"]["agglomerative"]["n_clusters"])

                elif name == "DBSCAN":
                     mlflow.log_param("eps",config["model"]["dbscan"]["eps"])

                mlflow.log_param("min_samples",config["model"]["dbscan"]["min_samples"])

                mlflow.log_metric("Silhouette",metrics["Silhouette"])

                mlflow.log_metric("Davies",metrics["Davies"])

                mlflow.log_metric("Calinski",metrics["Calinski"])

                mlflow.sklearn.log_model(model,name)

                if metrics["Silhouette"] > best_score:

                   best_score = metrics["Silhouette"]

                   best_model = model

                   best_labels = labels
  
        os.makedirs("artifacts/data",exist_ok=True)

        self.df.to_csv("artifacts/data/rfm_dataset.csv", index=False)

        joblib.dump(best_model,"artifacts/models/best_model.pkl")

        kmeans = KMeans(n_clusters=config["model"]["kmeans"]["n_clusters"],random_state=config["model"]["random_state"])

        kmeans_labels = kmeans.fit_predict(self.X)

        joblib.dump(kmeans, "artifacts/models/kmeans_model.pkl")

        self.df["Cluster"] = kmeans_labels

        cluster_profiles = self.df.groupby("Cluster").agg({"Recency": "mean","Frequency": "mean","Monetary": "mean","AverageOrderValue": "mean","CustomerValue": "mean"}).round(2)

        segment_names = { 0: "New Customers", 1: "VIP Customers", 2: "At Risk Customers", 3: "Inactive Customers"}

        cluster_profiles["Segment"] = cluster_profiles.index.map(segment_names)

        cluster_profiles.to_csv("artifacts/cluster_profiles.csv")

        self.df.to_csv("artifacts/rfm_clustered.csv",index=False)

        metrics_df = pd.DataFrame(results)

        metrics_df.to_csv("artifacts/model_metrics.csv",index=False)

        mlflow.log_artifact("artifacts/model_metrics.csv")
    
        mlflow.log_artifact("artifacts/rfm_clustered.csv")
   
        mlflow.sklearn.log_model(best_model,artifact_path="best_model")
       
        logger.info("Best Model Saved")

        return pd.DataFrame(results)

if __name__ == "__main__":
   print("Main block running")
   trainer =TrainModels()
   trainer.train()
