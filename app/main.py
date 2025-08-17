# app/main.py
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os
import joblib
import pandas as pd

app = FastAPI(title="API - Supervivencia Titanic", version="1.0.0")

# --- Esquema de entrada con validaciones Pydantic ---
class Passenger(BaseModel):
    pclass: Literal[1, 2, 3]
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=120)
    sibsp: int = Field(ge=0)
    parch: int = Field(ge=0)
    fare: float = Field(ge=0)
    embarked: Literal["C", "Q", "S"]

# --- Carga de artefacto (modelo + threshold + features) ---
MODEL_PATH = os.getenv("MODEL_PATH", "model/logistic_titanic_pipeline.pkl")
try:
    meta = joblib.load(MODEL_PATH)            # dict: {"model","threshold","features"}
    clf = meta["model"]
    THRESHOLD = float(meta.get("threshold", 0.5))
    FEATURES = meta.get("features", ['pclass','sex','age','sibsp','parch','fare','embarked'])
except Exception:
    clf = None
    THRESHOLD = 0.5
    FEATURES = ['pclass','sex','age','sibsp','parch','fare','embarked']

@app.get("/")
def home():
    return {"message": "API Titanic OK"}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": clf is not None}

@app.post("/predict")
def predict(
    passenger: Passenger = Body(
        ...,
        examples={
            "alta": {
                "summary": "Alta prob. de sobrevivir",
                "value": {
                    "pclass": 1, "sex": "female", "age": 22,
                    "sibsp": 0, "parch": 1, "fare": 80.0, "embarked": "C"
                }
            },
            "baja": {
                "summary": "Baja prob. de sobrevivir",
                "value": {
                    "pclass": 3, "sex": "male", "age": 30,
                    "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S"
                }
            }
        }
    ),
    confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0)
):
    if clf is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    try:
        df = pd.DataFrame([passenger.model_dump()])
        proba = float(clf.predict_proba(df[FEATURES])[:, 1][0])
        thr = confidence if confidence is not None else THRESHOLD
        label = int(proba >= thr)
        return {
            "survived": label,
            "probability": round(proba, 6),
            "confidence": thr
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")

