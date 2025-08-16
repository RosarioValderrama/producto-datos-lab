## Laboratorio — ML en Producción con Titanic
**Este laboratorio implementa un flujo completo de Machine Learning en producción usando el dataset Titanic (OpenML).**

Incluye:
- Entrenamiento con Pipeline de scikit-learn (imputación, codificación y modelo).
- Evaluación con F1-score y elección de umbral.
- Exportación de artefacto (modelo + umbral + features).
- API con FastAPI para servir predicciones.
- Cliente para consumir la API.
- CI/CD con GitHub Actions y pytest para monitorear el rendimiento (F1) sobre cohortes distintas (simulación de “futuro”).

*Inspirado en Introduction to ML in Production (DeepLearning.AI) y adaptado al caso Titanic.*

## Estructura del repositorio

```
producto-datos-lab/
├─ .github/workflows/
│  └─ producto-datos-lab.yml            # Workflow de CI (pytest)
├─ app/
│  ├─ data/
│  │  ├─ titanic_test_base.csv          # Cohorte base (holdout)
│  │  └─ titanic_test_future.csv        # Cohorte "futuro" (distribución distinta)
│  └─ test_titanic.py                   # Test unitario (F1 > umbral)
├─ model/
│  └─ logistic_titanic_pipeline.pkl     # Artefacto: pipeline + umbral + features
├─ notebooks/
│  ├─ 00_supervivencia_titanic.ipynb    # Entrenamiento + métricas + export artefacto
│  ├─ 01_server_titanic.ipynb           # API FastAPI (endpoint /predict)
│  ├─ 02_client_titanic.ipynb           # Cliente para consumir la API
│  └─ 03_predicciones_futuras.ipynb     # Generación de CSVs y monitoreo (drift)
├─ src/
│  ├─ dataset.py                        # Carga Titanic (OpenML/CSV)
│  └─ features.py                       # Utilidades p/ comparar distribuciones (KS-test)
├─ requirements.txt
└─ README.md
```

## Requisitos
**Python 3.11**
- pip 
- Git + cuenta de GitHub (para CI/CD)
### Prerequisito: Tener [conda](https://docs.conda.io/en/latest/) instalado en tu computador.
Vamos a usar Conda para construir un entorno virtual nuevo.

 
### 1. Creando el entorno virtual (Virtual Environment)

1) Crear y activar entorno
```bash
conda create -n producto-datos-lab python=3.11 -y
conda activate producto-datos-lab
```

2) Instalar dependencias
```bash
pip install -r requirements.txt
```

3) Registrar kernel de Jupyter
```bash
python -m ipykernel install --user --name producto-datos-lab
```

4) Abrir Jupyter Lab
```bash
jupyter lab
```

**Todo el trabajo que realicemos con este código será en este entorno. Así que al trabajar con estos archivos siempre tiene que estar activo el `producto-datos-lab`.**
 
### 2. Instalando las dependencias usando PIP 
 
Antes de seguir, verifica que en el terminal de Anaconda estés dentro del directorio `producto-datos-lab`, el cual incluye el archivo `requirements.txt`. Este archivo enlista todas las dependencias necesarias y podemos usarlo para instalarlas todas:
 
```bash
pip install -r requirements.txt
```
 
Este comando puede demorar un rato dependiendo de la velocidad del computador y la de la conexión a Internet. Una vez que termine ya está listo todo para comenzar una sesión de Jupyter Lab o Notebook.

Luego debemos enlazar el kernel de jupyter lab a nuestro nuevo enviroment:

```bash
python -m ipykernel install --user --name producto-datos-lab
```

 
### 3. Iniciando Jupyter Lab
 
Jupyter lab debería haber quedado instalado en el paso anterior, así que basta con escribir:

```bash
jupyter lab
```

### 4. Generando modelo de ML

**00 — Entrenamiento del modelo (Pipeline)**
*Notebook: notebooks/00_supervivencia_titanic.ipynb*

- Cargar dataset Titanic (OpenML).
- Hacer train/test split (estratificado).
- Definir Pipeline:
    - SimpleImputer (num=median, cat=most_frequent)
    - StandardScaler (num)
    - OneHotEncoder (cat, handle_unknown='ignore')
    - LogisticRegression(max_iter=1000)
    - Métricas: Accuracy, Precision, Recall, F1, ROC-AUC (umbral 0.5).
    -Búsqueda de umbral que maximiza F1.
    - Exportar artefacto a model/logistic_titanic_pipeline.pkl con el siguiente contenido:
```bash
{
  "model": <Pipeline>,
  "threshold": <float>,
  "features": ["pclass","sex","age","sibsp","parch","fare","embarked"]
}
```
**01 — Servidor (API FastAPI)**
*Notebook: notebooks/01_server_titanic.ipynb*
- Carga model/logistic_titanic_pipeline.pkl.

Endpoints:
GET / → información y umbral usado.
GET /healthz → healthcheck.
POST /predict → predicción.

Schema de entrada (/predict):
```bash
{
  "pclass": 1,
  "sex": "female",
  "age": 22,
  "sibsp": 0,
  "parch": 1,
  "fare": 80.0,
  "embarked": "C"
}
```
*Parámetro opcional ?confidence=0.55 para forzar umbral. Si no se envía, la API usa el umbral del artefacto.*

**02 — Cliente**
*Notebook: notebooks/02_client_titanic.ipynb*

- Configura BASE_URL = "http://127.0.0.1:8000".
- Envía JSON al endpoint /predict (ejemplos: Helene / Giles).

Incluye función batch que devuelve un DataFrame con prob_survive, pred_class, pred_label, threshold_used.

**03 — “Predicciones futuras” / Monitoreo**
*Notebook: notebooks/03_predicciones_futuras.ipynb*

Genera CSVs en app/data/:
- titanic_test_base.csv (holdout normal).
- titanic_test_future.csv (cohorte distinta, p. ej. solo 3ª clase).

*Usa src/features.py para comparar distribuciones (KS-test) entre cohortes y documentar drift.*

**Tests (pytest)**
*Archivo: app/test_titanic.py*

- Carga el artefacto y el CSV de test.
- Calcula probabilidades, aplica el umbral y asserta F1 > 0.70.

Ejecutar localmente:
```bash
cd app
pytest -q
```
*Para “futuro”, cambia el CSV del test a titanic_test_future.csv y vuelve a ejecutar.*


## 5. CI/CD con GitHub Actions
En este laboratorio usamos GitHub Actions para automatizar dos cosas clave:
- Pruebas automáticas cada vez que cambias código (o el modelo) — medimos F1 y fallamos el build si baja de un umbral.
- Monitoreo simple de desempeño sobre una cohorte “futura” para detectar cambios de distribución (drift).

**¿Qué es GitHub Actions?**
Es la plataforma de CI/CD integrada en GitHub que ejecuta workflows (archivos YAML) cuando ocurren eventos (push, PR, cron, etc.). En nuestro caso, el workflow:
- Instala dependencias.
- Carga tu artefacto (model/logistic_titanic_pipeline.pkl).
- Ejecuta pytest con un test que calcula F1 sobre un CSV de test.

**Fork y activación**
Haz fork de este repo en tu cuenta de GitHub.
En la pestaña Actions del fork, pulsa Enable workflows.
Clona tu fork localmente:

```bash
git clone https://github.com/<tu-usuario>/producto-datos-lab.git
cd producto-datos-lab
```

**Test unitario (pytest) para Titanic**
*Archivo: app/test_titanic.py.*
Este test carga el artefacto (pipeline + umbral + features), lee un CSV de test y asserta que F1 > 0.70. El dataset se pasa por variable de entorno para poder probar “base” y “futuro” con el mismo test.

# app/test_titanic.py
```bash
import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score

DATA = os.getenv("TEST_DATA_PATH", "app/data/titanic_test_base.csv")
ARTIFACT = os.getenv("MODEL_PATH", "model/logistic_titanic_pipeline.pkl")

def test_f1_over_threshold():
    bundle = joblib.load(ARTIFACT)
    pipe = bundle["model"]
    features = bundle["features"]
    threshold = float(bundle.get("threshold", 0.5))

    df = pd.read_csv(DATA)
    y = df["survived"].astype(int)
    X = df[features]

    proba = pipe.predict_proba(X)[:, 1]
    yhat = (proba >= threshold).astype(int)

    f1 = f1_score(y, yhat)
    assert f1 > 0.70, f"F1={f1:.3f} por debajo del umbral en {DATA}"
```
Cambia el umbral del assertion si lo necesitas.
Los CSV app/data/titanic_test_base.csv y app/data/titanic_test_future.csv se generan en el notebook 03_predicciones_futuras.ipynb.

**Workflow de CI (YAML)**
*Archivo: .github/workflows/producto-datos-lab.yml.*
Este workflow:
- Corre en Python 3.11.
- Cachea las dependencias para acelerar builds.
- Ejecuta pytest dos veces con una matrix: una para la cohorte “base” y otra para “futuro”.
- Se dispara en push/pull_request y semanalmente para monitoreo.

```bash
name: PD-MDS-Lab

on:
  push:
    paths:
      - 'app/**'
      - 'model/**'
      - '.github/workflows/**'
      - 'requirements.txt'
  pull_request:
    paths:
      - 'app/**'
      - 'model/**'
      - '.github/workflows/**'
      - 'requirements.txt'
  workflow_dispatch:
  schedule:
    - cron: '0 9 * * 1'  # Lunes 09:00 UTC

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        dataset:
          - app/data/titanic_test_base.csv
          - app/data/titanic_test_future.csv

    env:
      TEST_DATA_PATH: ${{ matrix.dataset }}
      MODEL_PATH: model/logistic_titanic_pipeline.pkl

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ runner.os }}-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            pip-${{ runner.os }}-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Pytest (F1 debe superar el umbral)
        run: |
          cd app
          pytest -q
```
*Si quieres que “futuro” no falle el build (sólo alerte), puedes separar en dos jobs y usar continue-on-error: true en el job futuro.*

**Empujar cambios y ver resultados**
Haz un cambio dentro de app/ o model/ (por ejemplo, comenta una línea del test).

git add --all && git commit -m "Test CI Titanic" && git push.

Abre Actions en tu repo y revisa el run: verás dos ejecuciones de pytest (base/futuro).

**Simular drift y caída de F1**

En notebooks/03_predicciones_futuras.ipynb generas titanic_test_future.csv con una distribución distinta (p. ej., solo 3ª clase). Si el F1 cae bajo el umbral, el job correspondiente fallará. Esto te permite:
- Documentar el cambio (usa src/features.py para KS-test).

- Ajustar umbral o reentrenar y volver a exportar el artefacto.

**Consejos y debugging**

“No module named …” en CI → confirma requirements.txt.

“File not found model/*.pkl” → exporta el modelo en el notebook 00 y haz commit del archivo en model/.

F1 bajo → baja el umbral solo si está justificado y documenta; de lo contrario, reentrena.


Badge en el README para mostrar el estado del CI:
![CI](https://github.com/RosarioValderrama/producto-datos-lab/actions/workflows/producto-datos-lab.yml/badge.svg)
