# app/main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal
import os, joblib, pandas as pd

app = FastAPI(title="API - Supervivencia Titanic", version="1.0.0")

class Passenger(BaseModel):
    pclass: Literal[1, 2, 3]
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=100)
    sibsp: int = Field(ge=0)
    parch: int = Field(ge=0)
    fare: float = Field(ge=0)
    embarked: Literal["C", "Q", "S"]

MODEL_PATH = os.getenv("MODEL_PATH", "model/logistic_titanic_pipeline.pkl")
try:
    meta = joblib.load(MODEL_PATH)
    clf = meta["model"]
    THRESHOLD = float(meta.get("threshold", 0.5))
    FEATURES = meta.get("features", ['pclass','sex','age','sibsp','parch','fare','embarked'])
except Exception as e:
    clf = None
    THRESHOLD, FEATURES = 0.5, ['pclass','sex','age','sibsp','parch','fare','embarked']

@app.get("/")
def home():
    return {"message": "API Titanic OK"}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": clf is not None}

@app.post("/predict")
def predict(passenger: Passenger, confidence: float | None = Query(default=None, ge=0.0, le=1.0)):
    if clf is None:
        raise HTTPException(500, detail="Modelo no cargado")
    try:
        df = pd.DataFrame([passenger.model_dump()])
        proba = float(clf.predict_proba(df[FEATURES])[:, 1][0])
        thr = confidence if confidence is not None else THRESHOLD
        label = int(proba >= thr)
        return {"survived": label, "probability": proba, "confidence": thr}
    except Exception as e:
        raise HTTPException(400, detail=f"Error en predicción: {e}")
