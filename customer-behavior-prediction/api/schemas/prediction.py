"""Pydantic schemas for prediction API (placeholder)."""

from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    CustomerID: int
    RecencyDays: float
    Frequency: float
    Monetary: float


class PredictResponse(BaseModel):
    CustomerID: int
    churn_probability: float
    churn_label: int
    note: str = "stub response"
