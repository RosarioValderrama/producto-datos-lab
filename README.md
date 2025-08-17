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
│   └── logistic_titanic_pipeline.pkl     # Artefacto principal
│   
├── notebooks/
│   ├── 00_supervivencia_titanic.ipynb    # Entrenamiento- preprocesamiento- métricas- artefacto
│   ├── 01_server_titanic.ipynb           # Versión notebook del servidor (pruebas locales con /docs)
│   ├── 02_client_titanic.ipynb           # llamar a la API (requests/cURL) con ejemplos de JSON
│   └── 03_predicciones_futuras.ipynb     # Genera y guarda los CSV de app/data 
│
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
**`notebooks/02_client_titanic.ipynb` — Cliente HTTP de pruebas**
- Envía múltiples requests a `POST /predict` (local o Render),
- Muestra payloads y respuestas (cliente externo)

---
**`notebooks/03_predicciones_futuras.ipynb` — Evaluación con datos “futuros”**
- Evalúa el modelo sobre el set “futuro” y compara métricas (F1 vs. el set base),
- Justifica el test de drift en CI (el F1 podría bajar en el conjunto futuro).
- Genera y guarda dos CSV dentro de `app/data/`:
  - `titanic_test_base.csv` → conjunto de prueba “estable” (similar al split original).
  - `titanic_test_future.csv` → conjunto “futuro” que simula drift (por ejemplo sesgo hacia 3ª clase).
- Estos archivos son consumidos por los tests de `pytest` en GitHub Actions.

app/data/
├─ titanic_test_base.csv
└─ titanic_test_future.csv



## 3. Ejecutar la API localmente
- Levanta el servidor FastAPI desde la raíz del repo con Uvicorn.
- Sirve para probar la API en la máquina antes de desplegarla.

Endpoints útiles:
- `GET /docs` → UI interactiva (Swagger) para probar `POST /predict`.
- `GET /healthz` → verificación rápida de que la app está viva.
Variable de entorno opcional: `MODEL_PATH` (por defecto `model/logistic_titanic_pipeline.pkl`).

**Esquema de entrada (qué espera el modelo)**
---
Desde la **raíz** del repositorio (asegúrate de activar tu entorno primero):

```bash
uvicorn app.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs
Healthcheck: http://127.0.0.1:8000/healthz

La API recibe un JSON con estos campos (validados con Pydantic):

- `pclass` (1, 2 o 3)  
- `sex` ("male" o "female")  
- `age` (0–120, float)  
- `sibsp`, `parch` (enteros ≥ 0)  
- `fare` (float ≥ 0)  
- `embarked` ("C", "Q" o "S")  

**Parámetro opcional de consulta**: `confidence` (0–1) para fijar el umbral de decisión.  
Si no se entrega, se usa el umbral guardado en el artefacto del modelo.

**Cómo hacer peticiones a la API:**
1. Swagger (recomendado):
Abre http://127.0.0.1:8000/docs
En POST /predict → Try it out → edita el JSON → Execute.

*Esquema de entrada (JSON)*
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

2. cURL / PowerShell / Python: puedes invocar `POST /predict` enviando el mismo JSON.

cURL (Bash)
```bash
curl -X POST "http://127.0.0.1:8000/predict?confidence=0.55" \
  -H "Content-Type: application/json" \
  -d '{
        "pclass": 1, "sex": "female", "age": 22,
        "sibsp": 0, "parch": 1, "fare": 80.0, "embarked": "C"
      }'
}
```

PowerShell (Windows)
```bash
$body = @{
  pclass=1; sex="female"; age=22; sibsp=0; parch=1; fare=80.0; embarked="C"
} | ConvertTo-Json

Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/predict?confidence=0.55" `
  -ContentType "application/json" -Body $body

```

Python (requests)
```bash
import requests

payload = {
  "pclass": 1, "sex": "female", "age": 22,
  "sibsp": 0, "parch": 1, "fare": 80.0, "embarked": "C"
}
r = requests.post(
  "http://127.0.0.1:8000/predict",
  json=payload,
  params={"confidence": 0.55},
  timeout=60
)
print(r.json())
```


**Tests y CI (qué corre y cuándo)**

- Qué hace el CI: en cada `push`/PR (y semanalmente) instala dependencias y ejecuta `pytest` sobre dos conjuntos:
  - Base (`app/data/titanic_test_base.csv`): debe superar el umbral de F1.
  - Future (`app/data/titanic_test_future.csv`): simula drift; puede fallar si el modelo degrada.
  *“future” se trata como monitoring check (no bloqueante) para detectar drift.*
- Para qué sirve: automatiza control de calidad y detecta cambios de distribución.
- Cómo se usa: haces un commit y miras la pestaña Actions en GitHub. Allí ves logs, F1 y si cada job pasó o falló.
- Si falla: revisa mensajes (faltan dependencias, no está el `.pkl`, F1 bajo). Ajusta umbral solo si está justificado; idealmente reentrena/mejora el modelo.



## 4. Workflows (CI/CD) con GitHub Actions
Automatiza pruebas del modelo en cada cambio y en una cohorte “futura” para detectar drift (caída de F1).

**Qué verifica el workflow**
*Archivo: .github/workflows/producto-datos-lab.yml.*

- Instala dependencias del repo.
- Carga el artefacto del modelo (`model/logistic_titanic_pipeline.pkl`).
- Ejecuta `pytest` sobre dos datasets:
  - Base: `app/data/titanic_test_base.csv` (debe superar el umbral de F1).
  - Future: `app/data/titanic_test_future.csv` (simula drift y puede fallar si el desempeño cae).
- Se ejecuta en:
  - `push` / `pull_request` (cuando cambian `app/**`, `model/**`, `.github/workflows/**`, `requirements.txt`)
  - Manual (`workflow_dispatch`)
  - Programado semanal (cron), si está configurado en el YAML.

**Requisitos previos**
- El archivo de modelo existe y está versionado: `model/logistic_titanic_pipeline.pkl`  
  (lo crea el notebook 00_supervivencia_titanic.ipynb).
- Están versionados los CSV de test en `app/data/` (los genera 03_predicciones_futuras.ipynb).
- `requirements.txt` incluye `pytest` (para que el runner pueda ejecutar tests).

**Simular drift y caída de F1**
En notebooks/03_predicciones_futuras.ipynb generas titanic_test_future.csv con una distribución distinta (solo 3ª clase). Si el F1 cae bajo el umbral, el job correspondiente fallará. Esto te permite:
- Documentar el cambio (usa src/features.py para KS-test).
- Ajustar umbral o reentrenar y volver a exportar el artefacto.

**Cómo activarlo en fork**
1. Haz Fork de este repo en la cuenta de GitHub.  
2. En el fork → pestaña Actions → pulsa Enable workflows.  
3. Clona el fork y sube un cambio (cualquier edición dentro de `app/` o `model/` sirve):
   ```bash
   git add --all
   git commit -m "Trigger CI (Titanic)"
   git push origin main

**Consejos y debugging**
“No module named …” en CI → confirma requirements.txt.
“File not found model/*.pkl” → exporta el modelo en el notebook 00 y haz commit del archivo en model/.
F1 bajo → baja el umbral solo si está justificado y documenta; de lo contrario, reentrena.

Badge en el README para mostrar el estado del CI:
![CI](https://github.com/RosarioValderrama/producto-datos-lab/actions/workflows/producto-datos-lab.yml/badge.svg)



## 5. Despliegue en Render (Web Service)
Exponer públicamente la API (`app/main.py`) para que se pueda consumir desde Internet.

https://producto-datos-lab-1.onrender.com

#### Endpoints.

Swagger UI: https://producto-datos-lab-1.onrender.com/docs

Healthcheck: https://producto-datos-lab-1.onrender.com/healthz

Predict (POST): https://producto-datos-lab-1.onrender.com/predict
(opcional ?confidence=0.55)

**Requisitos previos**
- El archivo de modelo está commiteado en el repo:
  - `model/logistic_titanic_pipeline.pkl` (dict con `{"model","threshold","features"}`).
- `requirements.txt` incluye al menos: `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `pydantic`.
- El módulo de la app existe en: `app/main.py` y define `app = FastAPI(...)`.

**Crear el servicio en Render**
1. Entra a <https://render.com> → **New** → **Web Service** → **Connect** tu repo.
2. Configura:
   - Branch: `main`
   - Root Directory: (deja vacío si tu `requirements.txt` está en la raíz)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Region/Plan: a tu elección (Free funciona para pruebas)
3. Environment Variables (opcional):
   - `MODEL_PATH = model/logistic_titanic_pipeline.pkl`  
     *(usa la ruta relativa al repo; si cambias la estructura, actualiza aquí)*

**Health check y verificación**
- En settings del servicio, usa Health Check Path: `/healthz`.
- Tras el deploy:
  - Abre `https://<tu-servicio>.onrender.com/healthz` → debe responder `{"status":"ok"}`.
  - Abre `https://<tu-servicio>.onrender.com/docs` → prueba `POST /predict` con un JSON válido.

> **Tip:** si ves errores tipo `ModuleNotFoundError` o `ImportError`, revisa que el paquete esté en `requirements.txt`.  
> Si aparece `FileNotFoundError` para el modelo, confirma que `model/logistic_titanic_pipeline.pkl` está en el repo y que `MODEL_PATH` apunta a esa ruta.

**Buenas prácticas**
- Activa **Auto-Deploy** en Render para que cada `push` a `main` redeploye tu API.
- Agrega la **URL pública** en la parte superior del README:
  - `**URL de la API:** https://<tu-servicio>.onrender.com`




### ANEXOS - Ejemplos rápidos de payload
> Más ejemplos y llamadas listas para copiar → ver **docs/payloads.md**.

