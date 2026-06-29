from fastapi import FastAPI
from schemas import CustomerFeatures
from predict import predict_cluster

app = FastAPI(title="Online Retail Customer Segmentation API", version="1.0")

@app.get("/")
def home():
    return {"message": "Online Retail API Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer:
CustomerFeatures):

    cluster = predict_cluster(customer.dict())

    return {"cluster": cluster}