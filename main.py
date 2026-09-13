from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="House Price Prediction API")

model_bundle = joblib.load("bagging_model.pkl")
if not isinstance(model_bundle, dict):
    model_bundle = {"bagging": model_bundle}

class HouseRequest(BaseModel):
    dientich: float = Field(gt=0)
    so_phong_ngu: int = Field(ge=0)
    method: Literal["bagging", "boosting", "stacking"] = "bagging"

@app.get("/")
def home():
    return {"message": "API Dự đoán giá nhà đang hoạt động!"}

@app.post("/predict")
def predict(data: HouseRequest):
    model = model_bundle.get(data.method)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{data.method}' chưa được train.",
        )

    input_data = np.array([[data.dientich, data.so_phong_ngu]])
    prediction = model.predict(input_data)
    return {
        "dientich": data.dientich,
        "so_phong_ngu": data.so_phong_ngu,
        "method": data.method,
        "gia_du_doan_ty": round(float(prediction[0]), 2)
    }