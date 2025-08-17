import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score

MODEL_PATH = os.getenv("MODEL_PATH", "model/logistic_titanic_pipeline.pkl")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "app/data/titanic_test_future.csv")

DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
FEATURES_FALLBACK = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
TARGET = "survived"

def load_model_and_meta():
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        clf = bundle["model"]
        thr = float(bundle.get("threshold", DEFAULT_THRESHOLD))
        features = bundle.get("features", FEATURES_FALLBACK)
        return clf, thr, features
    clf = bundle
    thr = DEFAULT_THRESHOLD
    features = FEATURES_FALLBACK
    return clf, thr, features

def test_f1_over_threshold():
    df = pd.read_csv(TEST_DATA_PATH)
    clf, thr, features = load_model_and_meta()

    X = df[features]
    y = df[TARGET].astype(int)

    proba = clf.predict_proba(X)[:, 1]
    yhat = (proba >= thr).astype(int)

    f1 = f1_score(y, yhat)
    assert f1 > 0.70, f"F1={f1:.3f} por debajo del umbral en {TEST_DATA_PATH}"
