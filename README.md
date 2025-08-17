## Laboratorio — ML en Producción / API – Supervivencia Titanic
Este laboratorio implementa un clasificador binario de supervivencia usando **scikit-learn** y **FastAPI**, con pipeline de preprocesamiento y métricas **F1** en CI (GitHub Actions).



## 1. Estructura del repositorio

producto-datos-lab/
├── .github/
│   └── workflows/
│       └── producto-datos-lab.yml        # Workflow de GitHub Actions: instala deps y ejecuta pytest 
├── app/
│   ├── data/
│   │   ├── titanic_test_base.csv         # Muestra base para validar el modelo (CI: “caso estable”)
│   │   └── titanic_test_future.csv       # Muestra “futura” para simular drift (CI: puede bajar el F1)
│   ├── test/
│   │   ├── test_titanic.py               # Test de F1 mínimo usando el set base (debe pasar)
│   │   └── test_titanic_future.py        # Test sobre set futuro (puede fallar si hay drift)
│   └── main.py                           # API FastAPI: carga el artefacto, expone /healthz y /predict
├── docs/
│   └── pruebas_estadisticas.md           
├── model/
│   ├── logistic_titanic_pipeline.pkl     # Artefacto principal
│   └── logistic_titanic_meta.pkl         # Artefacto alternativo
├── notebooks/
│   ├── 00_supervivencia_titanic.ipynb    # Entrenamiento- preprocesamiento- métricas- artefacto
│   ├── 01_server_titanic.ipynb           # Versión notebook del servidor (pruebas locales con /docs)
│   ├── 02_client_titanic.ipynb           # llamar a la API (requests/cURL) con ejemplos de JSON
│   ├── 03_predicciones_futuras.ipynb     # Simulación de predicciones futuras / drift
│   └── 04_generar_datos_test_CI.ipynb    # Genera y guarda los CSV de app/data 
├── requirements.txt                      # Dependencias 
└── README.md                             # Guía del proyecto



## 2. Requisitos e instalación
Recomendado Python 3.11.

```bash
# crear y activar entorno (ejemplo con venv)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# instalar dependencias
pip install -r requirements.txt
```

**Prerequisito:** Tener [conda](https://docs.conda.io/en/latest/) instalado en tu computador.
Usaremos Conda para aislar dependencias en un entorno virtual.

 **Crear y activar el entorno virtual (Virtual Environment)**

```bash
conda create -n producto-datos-lab python=3.11 -y
conda activate producto-datos-lab
```

 **Instalar dependencias**
```bash
pip install -r requirements.txt
```

 **Registrar kernel de Jupyter**
```bash
python -m ipykernel install --user --name producto-datos-lab
```

 **Iniciar Jupyter Lab**
```bash
jupyter lab
```

**Todo el trabajo que realicemos con este código será en este entorno. Así que al trabajar con estos archivos siempre tiene que estar activo el `producto-datos-lab`.**



### 2. Entrenamiento y archivo de modelo (artefacto)
> En esta guía usamos “artefacto” como sinónimo de archivo de modelo serializado.  
> Es el `.pkl` que guarda el Pipeline entrenado (preprocesamiento + clasificador) y sus metadatos(features esperadas y umbral).

Salidas esperadas tras entrenar:
model/
├─ logistic_titanic_pipeline.pkl # artefacto principal: dict con {"model", "threshold", "features"}
└─ logistic_titanic_meta.pkl # metadatos por separado; se puede omitir

---
**`notebooks/00_supervivencia_titanic.ipynb` — Entrenar y exportar el modelo**
- Prepara el dataset del Titanic, separa train/test y construye un Pipeline de scikit-learn con:
  - imputación de nulos (num/cat),
  - codificación One-Hot para categóricas,
  - estandarización de numéricas,
  - Regresión Logística como clasificador.
- Calcula métricas (F1-score) y busca umbral por F1.
- Exporta el artefacto en `model/logistic_titanic_pipeline.pkl` como un diccionario con:
  - `model`: el Pipeline entrenado (tiene `.predict_proba`)
  - `threshold`: umbral recomendado (float)
  - `features`: columnas de entrada esperadas (en orden)

> Tras ejecutar todo el notebook, verifica que existe `model/logistic_titanic_pipeline.pkl`.

---
**`notebooks/01_server_titanic.ipynb` — Prototipo de servidor (FastAPI)**
- Demuestra cómo cargar el artefacto y exponer:
  - `GET /` (home), `GET /healthz` (health check),
  - `POST /predict` (recibe el JSON del pasajero y devuelve probabilidad + etiqueta).
- Valida entradas con Pydantic (rangos y dominios).
- Este notebook es la base del archivo de aplicación que se usa en despliegue:
  - el código “de verdad” que levanta el servidor vive en `app/main.py`,
  - para correr local/Render se usa:  
    `uvicorn app.main:app --host 0.0.0.0 --port 8000`

---
**`notebooks/04_generar_datos_test_CI.ipynb` — Datasets de test para CI**
- Genera y guarda dos CSV dentro de `app/data/`:
  - `titanic_test_base.csv` → conjunto de prueba “estable” (similar al split original).
  - `titanic_test_future.csv` → conjunto “futuro” que simula drift (por ejemplo sesgo hacia 3ª clase).
- Estos archivos son consumidos por los tests de `pytest` en GitHub Actions.

app/data/
├─ titanic_test_base.csv
└─ titanic_test_future.csv

---
**`notebooks/02_client_titanic.ipynb` — Cliente HTTP de pruebas**
- Envía múltiples requests a `POST /predict` (local o Render),
- Muestra payloads y respuestas (cliente externo)

---
**`notebooks/03_predicciones_futuras.ipynb` — Evaluación con datos “futuros”**
- Evalúa el modelo sobre el set “futuro” y compara métricas (F1 vs. el set base),
- Justifica el test de drift en CI (el F1 podría bajar en el conjunto futuro).

---




























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
