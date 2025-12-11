# train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data_utils import load_config, load_raw_dataset
from model_utils import save_model


def train():
    config = load_config("local.variables.json")
    X, y, feature_cols = load_raw_dataset(config)

    print("Distribusi label:")
    print(y.value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config["random_state"]
    )

    print("\nTraining Random Forest (RAW model)...")

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        class_weight="balanced",
        random_state=config["random_state"],
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== AKURASI MODEL: {acc:.2%} ===\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix (Akurasi: {acc:.2%})")
    plt.show()

    # Feature importance
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(feature_cols)), importances[idx])
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in idx], rotation=45)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    save_model(model, feature_cols, config)
    print("\n[MODEL SAVED]")


if __name__ == "__main__":
    train()