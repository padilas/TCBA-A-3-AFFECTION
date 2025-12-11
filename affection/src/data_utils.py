# data_utils.py
import json
import pandas as pd
import numpy as np

def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)

def load_raw_dataset(config):
    df = pd.read_csv(config["input_csv_path"])

    df["ACC_Mag"] = np.sqrt(df["ACC_X"]**2 + df["ACC_Y"]**2 + df["ACC_Z"]**2)

    feature_cols = ["ACC_X", "ACC_Y", "ACC_Z", "BVP", "EDA", "TEMP", "ACC_Mag"]
    X = df[feature_cols]
    y = df["label"]

    return X, y, feature_cols