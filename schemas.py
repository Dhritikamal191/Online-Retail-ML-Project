from pydantic import BaseModel

class CustomerFeatures(BaseModel):
     Recency: float
     Frequency: float
     Monetary: float
     AverageOrderValue: float
     CustomerValue: float