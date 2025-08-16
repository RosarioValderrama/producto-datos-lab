import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score

# Paths por defecto (se pueden sobreescribir con variables de entorno)
MODEL_PATH = os.environ.get("MODEL_PATH", "../model/logistic_titanic_pipeline.pkl")
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "data/titanic_test_base.csv")

# Columnas que usa tu pipeline
FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
TARGET = 'survived'  # asegúrate que esta columna exista en tus CSVs

def test_f1_over_threshold():
    # Cargar datos de prueba
    df = pd.read_csv(TEST_DATA_PATH)

    # X e y
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Cargar modelo
    clf = joblib.load(MODEL_PATH)

    # Predicción y F1
    proba = clf.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    f1 = f1_score(y, y_pred)

    # Umbral razonable para titanic (ajústalo si gustas)
    assert f1 > 0.65, f"F1 demasiado bajo: {f1:.3f}"
