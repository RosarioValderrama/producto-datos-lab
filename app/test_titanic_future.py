import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score

MODEL_PATH = os.environ.get("MODEL_PATH", "../model/logistic_titanic_pipeline.pkl")
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "data/titanic_test_future.csv")

FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
TARGET = 'survived'

def test_f1_future_may_drop_but_runs():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    clf = joblib.load(MODEL_PATH)
    proba = clf.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    f1 = f1_score(y, y_pred)

    # Aquí no exigimos >0.65; sólo ver que corre y no es trivial
    assert f1 > 0.5, f"F1 muy bajo en futuro: {f1:.3f}"
