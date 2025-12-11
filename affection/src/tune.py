import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from data_utils import load_config
from model_utils import save_model

config = load_config("local.variables.json")

df = pd.read_csv(config["input_csv_path"])
df['ACC_Mag'] = np.sqrt(df['ACC_X']**2 + df['ACC_Y']**2 + df['ACC_Z']**2)

feature_cols = ["ACC_X", "ACC_Y", "ACC_Z", "BVP", "EDA", "TEMP", "ACC_Mag"]
X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=config["random_state"],
    stratify=y
)

print("Distribusi label:")
print(y.value_counts())
print("\nMulai tuning model...\n")

param_grid = {
    "n_estimators": [30, 50, 100],
    "max_depth": [1, 2, 3, 4],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
}

base_model = RandomForestClassifier(
    random_state=config["random_state"],
    n_jobs=-1
)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=4,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\n===== HASIL GRID SEARCH =====")
print("Best Params:", grid.best_params_)
print("Best CV Score (F1 Macro):", grid.best_score_)

best_model = grid.best_estimator_
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
gap = train_acc - test_acc

print("\n===== CEK OVERFIT / UNDERFIT =====")
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy : {test_acc:.3f}")
print(f"Gap           : {gap:.3f}")

if gap > 0.10:
    print("Model OVERFIT")
elif gap < -0.05:
    print("Model UNDERFIT")
else:
    print("Model stabil")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, test_pred))

cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Best Model")
plt.xlabel("Prediksi")
plt.ylabel("Actual")
plt.show()

results = pd.DataFrame(grid.cv_results_)
os.makedirs("outputs/metrics", exist_ok=True)

grid_csv_path = "outputs/metrics/gridsearch_results.csv"
results.to_csv(grid_csv_path, index=False)
print(f"[SAVE] Grid search results -> {grid_csv_path}")

metrics_path = config["metrics_output_path"]

with open(metrics_path, "w") as f:
    json.dump({
        "best_params": grid.best_params_,
        "best_cv_f1_macro": grid.best_score_,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "gap": gap
    }, f, indent=4)

print(f"[SAVE] Metrics -> {metrics_path}")
save_model(best_model, feature_cols, config)
print(f"[SAVE] Final Model Saved -> {config['model_output_path']}")