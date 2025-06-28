from fastapi import FastAPI, HTTPException
from pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc
import numpy as np

app = FastAPI()


model = mlflow.pyfunc.load_model("models:/GradientBoosting/Production")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try:

        input_df = data.dict()
        import pandas as pd
        df = pd.DataFrame([input_df])
        
        risk_prob = model.predict_proba(df)[:, 1][0]
        
        return PredictionResponse(risk_probability=risk_prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
