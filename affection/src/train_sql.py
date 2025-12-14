import argparse
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main(input_dir: str, model_dir: str, features_dir: str, metrics_dir: str):
    # === LOAD DATA ===
    input_file = os.path.join(input_dir, "preprocessed.csv")
    df = pd.read_csv(input_file)

    # === SAMA DENGAN train.py ===
    drop_cols = [c for c in ["ID", "LABEL"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["LABEL"]

    feature_cols = list(X.columns)

    # === TRAIN / TEST SPLIT ===
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # === MODEL (IDENTIK DENGAN train.py) ===
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # === EVALUATION ===
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # === SAVE MODEL ===
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_wesad.joblib")
    joblib.dump(model, model_path)

    # === SAVE FEATURES ===
    os.makedirs(features_dir, exist_ok=True)
    features_path = os.path.join(features_dir, "features.json")
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)

    # === SAVE METRICS ===
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "classification_report": report
            },
            f,
            indent=2
        )

    print("[DONE] Training finished")
    print("[INFO] Accuracy:", acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--features_output", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    args = parser.parse_args()

    main(
        args.input_data,
        args.model_output,
        args.features_output,
        args.metrics_output
    )