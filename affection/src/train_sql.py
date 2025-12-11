import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def main(input_dir: str, model_dir: str, features_dir: str, metrics_dir: str):
    input_file = os.path.join(input_dir, "preprocessed.csv")
    df = pd.read_csv(input_file)

    X = df.drop(columns=["LABEL"])
    y = df["LABEL"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(clf, model_path)

    os.makedirs(features_dir, exist_ok=True)
    features_path = os.path.join(features_dir, "features.txt")
    with open(features_path, "w") as f:
        f.write(",".join(X.columns))

    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.txt")
    acc = clf.score(X, y)
    with open(metrics_path, "w") as f:
        f.write(f"Training accuracy: {acc:.4f}")

    print(f"[SAVE] Model -> {model_path}")
    print(f"[SAVE] Features -> {features_path}")
    print(f"[SAVE] Metrics -> {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True,
                        help="Folder path with preprocessed data")
    parser.add_argument("--model_output", type=str, required=True,
                        help="Folder path to save model")
    parser.add_argument("--features_output", type=str, required=True,
                        help="Folder path to save features")
    parser.add_argument("--metrics_output", type=str, required=True,
                        help="Folder path to save metrics")
    args = parser.parse_args()

    main(args.input_data, args.model_output, args.features_output, args.metrics_output)