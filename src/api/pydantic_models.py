from pydantic import BaseModel
from typing import List, Union

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    std_amount: float
    transaction_count: int
    fraud_count: int
    ProductCategory_financial_services: int
    ProductCategory_other: int
    ChannelId_1: int
    ChannelId_2: int
    ChannelId_3: int
    CurrencyCode_USD: int
    CurrencyCode_UGX: int


class PredictionResponse(BaseModel):
    risk_probability: float
