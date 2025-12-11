# score.py
import json
import numpy as np
import pandas as pd
import traceback
import pyodbc
import argparse
import os

from src.model_utils import load_model
from src.data_utils import load_config

model = None
feature_names = None
config = None


def init():
    """
    Azure ML akan memanggil init() sekali saat container start.
    Fokus: load model & config, jangan koneksi DB di sini.
    """
    global model, feature_names, config

    # load config & model
    config = load_config("local.variables.json")
    model, feature_names = load_model(config)

    print("[READY] Model loaded")


def run(raw):
    """
    Azure ML akan memanggil run() setiap ada request.
    Di sini baru boleh connect ke DB kalau perlu.
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

        # optional: connect ke SQL hanya kalau perlu logging
        try:
            conn = pyodbc.connect(
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER=tcp:{config['sql_server']},1433;"
                f"DATABASE={config['sql_database']};"
                f"UID={config['sql_username']};"
                f"PWD={config['sql_password']};"
                "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
            )
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Predictions (ACC_X, ACC_Y, ACC_Z, BVP, EDA, TEMP, Prediction) VALUES (?, ?, ?, ?, ?, ?, ?)",
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


# CLI mode untuk testing lokal
def json_mode(path):
    with open(path, "r") as f:
        data = f.read()
    output = run(data)
    print("\n=== PREDIKSI ===")
    print(output)


def manual_mode():
    print("\n=== MANUAL INPUT MODE ===")
    sample = {}
    for key in ["ACC_X", "ACC_Y", "ACC_Z", "BVP", "EDA", "TEMP"]:
        sample[key] = float(input(f"{key}: "))
    output = run(json.dumps(sample))
    print("\n=== PREDIKSI ===")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Path JSON input file")
    parser.add_argument("--manual", action="store_true", help="Manual input mode")
    args = parser.parse_args()

    init()

    if args.json:
        json_mode(args.json)
    elif args.manual:
        manual_mode()
    else:
        print("\nSCORE SERVICE READY")
        print("Gunakan salah satu:")
        print("  python src/score.py --json sample.json")
        print("  python src/score.py --manual")