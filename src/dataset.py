import pandas as pd
from sklearn.datasets import fetch_openml
from typing import Tuple, Optional

FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

def load_titanic(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga el dataset Titanic desde OpenML y devuelve X (features crudas) e y (0/1).
    """
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    y = y.astype(int)
    return X[FEATURES], y

def load_test_csv(path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Lee un CSV con las columnas crudas del pipeline y, si está, la columna 'survived'.
    """
    df = pd.read_csv(path)
    X = df[FEATURES].copy()
    y = df['survived'].astype(int) if 'survived' in df.columns else None
    return X, y
