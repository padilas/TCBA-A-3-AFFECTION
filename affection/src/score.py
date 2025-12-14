import os
import json
import numpy as np
import pandas as pd
import traceback
import joblib

model = None
feature_names = None


def init():
    global model, feature_names

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if model_dir is None:
        raise RuntimeError("AZUREML_MODEL_DIR tidak ditemukan.")

    # Try multiple artifact filename patterns
    model_candidates = ["rf_wesad.joblib", "model.pkl", "model.joblib"]
    features_candidates = ["features.json", "features.txt"]

    model_path = None
    for fname in model_candidates:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        model_path = os.path.join(model_dir, model_candidates[0])

    features_path = None
    for fname in features_candidates:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            features_path = p
            break
    if features_path is None:
        features_path = os.path.join(model_dir, features_candidates[0])

    model = joblib.load(model_path)

    try:
        with open(features_path, "r") as f:
            content = f.read()
            if features_path.endswith(".json"):
                feature_names = json.loads(content)
            else:
                feature_names = content.strip().split(",")
    except Exception:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None:
            feature_names = ["ACC_X", "ACC_Y", "ACC_Z", "BVP", "EDA", "TEMP", "ACC_Mag"]

    print("[READY] Model & features loaded")


def run(raw):
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

        sample["ACC_Mag"] = np.sqrt(
            sample["ACC_X"]**2 +
            sample["ACC_Y"]**2 +
            sample["ACC_Z"]**2
        )

        base_df = pd.DataFrame([sample])

        expected = feature_names
        if expected is None:
            expected = list(base_df.columns)

        for col in expected:
            if col not in base_df.columns:
                base_df[col] = 0.0

        df = base_df[expected]
        pred = int(model.predict(df)[0])

        return json.dumps({"prediction": pred})

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "trace": traceback.format_exc()
        })