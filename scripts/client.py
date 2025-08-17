# scripts/client.py
import os, json, time, pathlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = os.getenv("BASE_URL", "https://producto-datos-lab-1.onrender.com")  # cambia si quieres probar local

def session_with_retries():
    s = requests.Session()
    retry = Retry(
        total=6, backoff_factor=0.7,
        status_forcelist=(502, 503, 504),
        allowed_methods=("GET", "POST")
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def main():
    s = session_with_retries()

    print(f"[INFO] Usando BASE_URL = {BASE}")

    # 1) Wake up /healthz
    try:
        r = s.get(f"{BASE}/healthz", timeout=20)
        print("[healthz]", r.status_code, r.text)
    except Exception as e:
        print("[healthz] ERROR:", e)

    # 2) Tres payloads distintos
    payloads = [
        {
            "name": "Caso 1 - mujer 1a clase",
            "json": {"pclass": 1, "sex": "female", "age": 22, "sibsp": 0, "parch": 1, "fare": 80.0, "embarked": "C"},
            "params": {}
        },
        {
            "name": "Caso 2 - hombre 3a clase (umbral 0.60)",
            "json": {"pclass": 3, "sex": "male", "age": 35, "sibsp": 0, "parch": 0, "fare": 8.05, "embarked": "S"},
            "params": {"confidence": 0.60}
        },
        {
            "name": "Caso 3 - mujer 2a clase (umbral 0.40)",
            "json": {"pclass": 2, "sex": "female", "age": 28, "sibsp": 1, "parch": 0, "fare": 26.0, "embarked": "Q"},
            "params": {"confidence": 0.40}
        },
    ]

    results = []
    for i, p in enumerate(payloads, start=1):
        try:
            r = s.post(f"{BASE}/predict", json=p["json"], params=p["params"], timeout=30)
            try:
                body = r.json()
            except Exception:
                body = {"raw_text": r.text}

            print(f"[predict #{i}] {p['name']}: {r.status_code} -> {body}")

            results.append({
                "name": p["name"],
                "url": f"{BASE}/predict",
                "sent_json": p["json"],
                "sent_params": p["params"],
                "status_code": r.status_code,
                "response": body,
            })
        except Exception as e:
            print(f"[predict #{i}] ERROR:", e)
            results.append({
                "name": p["name"],
                "error": str(e),
            })
        time.sleep(0.2)

    # 3) Guardar evidencia
    out_dir = pathlib.Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "client_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Evidencia guardada en {out_path.resolve()}")

if __name__ == "__main__":
    main()
