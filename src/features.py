import pandas as pd
from typing import List
from scipy import stats

FEATURES: List[str] = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

def _to_numeric_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte categorías a códigos numéricos para poder usar KS-test.
    (Esto es solo para monitoreo; el pipeline real sigue usando categorías crudas.)
    """
    x = df.copy()[FEATURES]
    # map categóricas a códigos consistentes
    x['sex'] = x['sex'].map({'male': 1, 'female': 0}).astype('float32')
    x['embarked'] = x['embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype('float32')
    # asegurar numéricas
    for col in ['pclass', 'age', 'sibsp', 'parch', 'fare']:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    return x

def compare_distributions(dfa: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
    """
    Compara distribuciones (KS-test) entre dos cohortes (base vs “futuro”).
    Devuelve un DataFrame con statistic y p_value por feature.
    """
    a = _to_numeric_for_drift(dfa)
    b = _to_numeric_for_drift(dfb)

    rows = []
    for col in a.columns:
        aa = a[col].dropna()
        bb = b[col].dropna()
        if aa.empty or bb.empty:
            stat, p = float('nan'), float('nan')
        else:
            stat, p = stats.ks_2samp(aa, bb)
        rows.append({'feature': col, 'statistic': stat, 'p_value': p})
    return pd.DataFrame(rows).sort_values('p_value')
