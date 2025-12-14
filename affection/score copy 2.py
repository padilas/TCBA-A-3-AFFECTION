# score.py
import os
import json
import numpy as np
import pandas as pd
import traceback
import joblib

# optional: hanya kalau environment punya pyodbc
try:
    import pyodbc
except ImportError:
    pyodbc = None

model = None
feature_names = None

def init():
    """
    Dipanggil sekali saat container start.
    Fokus: load model & feature list dari artefak yang dimount Azure ML.
    """
    global model, feature_names

    # Path mount model dari Azure ML
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if model_dir is None:
        raise RuntimeError("AZUREML_MODEL_DIR tidak ditemukan. Pastikan model ter‑register.")

    # load model.pkl
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)

    # load features.txt kalau ada
    features_file = os.path.join(model_dir, "features.txt")
    if os.path.exists(features_file):
        with open(features_file, "r") as f:
            feature_names = f.read().split(",")
    else:
        # fallback default
        feature_names = ["ACC_X", "ACC_Y", "ACC_Z", "BVP", "EDA", "TEMP", "ACC_Mag"]

    print("[READY] Model loaded from", model_path)


def run(raw):
    """
    Dipanggil setiap ada request ke endpoint.
    Input: JSON string atau dict dengan sensor values.
    Output: JSON string dengan prediksi.
    """
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            raw = json.loads(raw)

        sample = {
            "ACC_X": float(raw["ACC_X"]),
            "ACC_Y": float(raw["ACC_Y"]),
            "ACC_Z": float(raw["ACC_Z"]),
            "BVP": float(raw["BVP"]),
            "EDA": float(raw["EDA"]),
            "TEMP": float(raw["TEMP"]),
        }

        # fitur tambahan
        sample["ACC_Mag"] = np.sqrt(
            sample["ACC_X"]**2 + sample["ACC_Y"]**2 + sample["ACC_Z"]**2
        )

        df = pd.DataFrame([sample])[feature_names]
        pred = int(model.predict(df)[0])

        # optional logging ke SQL
        if pyodbc is not None:
            try:
                conn = pyodbc.connect(
                    "DRIVER={ODBC Driver 17 for SQL Server};"
                    f"SERVER=tcp:{os.getenv('SQL_SERVER')},1433;"
                    f"DATABASE={os.getenv('SQL_DATABASE')};"
                    f"UID={os.getenv('SQL_USERNAME')};"
                    f"PWD={os.getenv('SQL_PASSWORD')};"
                    "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
                )
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO TempAffection (ACC_X, ACC_Y, ACC_Z, BVP, EDA, TEMP, LABEL) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    sample["ACC_X"], sample["ACC_Y"], sample["ACC_Z"],
                    sample["BVP"], sample["EDA"], sample["TEMP"], pred
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as db_err:
                print(f"[WARN] DB logging skipped: {db_err}")

        return json.dumps({"prediction": pred})

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "trace": traceback.format_exc()
        })