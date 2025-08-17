# app/tests/test_titanic.py
import os, joblib, pandas as pd
from sklearn.metrics import f1_score

MODEL_PATH = os.environ.get("MODEL_PATH", "../model/logistic_titanic_pipeline.pkl")
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "data/titanic_test_future.csv")

FEATURES = ['pclass','sex','age','sibsp','parch','fare','embarked']
TARGET = 'survived'

def test_f1_over_threshold():
    df = pd.read_csv(TEST_DATA_PATH)
    X, y = df[FEATURES], df[TARGET].astype(int)

    meta = joblib.load(MODEL_PATH)       # ← cargamos el dict meta
    clf = meta["model"]

    proba = clf.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    f1 = f1_score(y, y_pred)
    assert f1 > 0.55, f"F1 demasiado bajo: {f1:.3f}"