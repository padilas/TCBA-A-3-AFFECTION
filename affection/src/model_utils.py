# model_utils.py
import json
import joblib

def save_model(model, feature_names, config):
    joblib.dump(model, config["model_output_path"])
    with open(config["features_output_path"], "w") as f:
        json.dump(feature_names, f, indent=4)

def load_model(config):
    model = joblib.load(config["model_output_path"])
    with open(config["features_output_path"], "r") as f:
        feature_names = json.load(f)
    return model, feature_names