# Payloads de ejemplo (Titanic)

## JSON listos para /predict

**Ejemplo A — alta prob. de sobrevivir**
```json
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

**Ejemplo B — baja prob. de sobrevivir**
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 30,
  "sibsp": 0,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```


**Llamadas listas para copiar**

cURL (Bash)

```bash
curl -X POST "http://127.0.0.1:8000/predict?confidence=0.55" \
  -H "Content-Type: application/json" \
  -d '{
        "pclass": 1, "sex": "female", "age": 22,
        "sibsp": 0, "parch": 1, "fare": 80.0, "embarked": "C"
      }'
```

PowerShell (Windows)

```shell
$body = @{
  pclass=1; sex="female"; age=22; sibsp=0; parch=1; fare=80.0; embarked="C"
} | ConvertTo-Json

Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/predict?confidence=0.55" `
  -ContentType "application/json" -Body $body
```

Python (requests)
```python

import requests

payload = {"pclass":1,"sex":"female","age":22,"sibsp":0,"parch":1,"fare":80.0,"embarked":"C"}
r = requests.post("http://127.0.0.1:8000/predict", json=payload, params={"confidence":0.55})
print(r.json())
```