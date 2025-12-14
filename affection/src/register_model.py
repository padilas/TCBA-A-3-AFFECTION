import argparse
import os
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

def _first_existing(base_dir, candidates):
    for name in candidates:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    return os.path.join(base_dir, candidates[0])


def main(model_path: str, features_path: str, metrics_path: str, output_file: str):
    try:
        ml_client = MLClient.from_config(DefaultAzureCredential())
    except Exception:
        ml_client = MLClient.from_config(InteractiveBrowserCredential())

    # Support multiple artifact filenames
    model_file = _first_existing(model_path, ["rf_wesad.joblib"])
    features_file = _first_existing(features_path, ["features.json"]) 
    metrics_file = _first_existing(metrics_path, ["metrics.json"]) 

    model = Model(
        path=model_file,
        name="affection-rf-model",
        type="custom_model",
        description="Random Forest retrained from SQL pipeline",
        tags={
            "source": "pipeline",
            "features_path": features_file,
            "metrics_path": metrics_file
        }
    )
    registered_model = ml_client.models.create_or_update(model)

    with open(output_file, "w") as f:
        f.write(registered_model.id)

    print(f"[REGISTERED] Model ID -> {registered_model.id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--registered_model_info", type=str, required=True,
                        help="File path to save model ID")
    args = parser.parse_args()

    main(args.model_path, args.features_path, args.metrics_path, args.registered_model_info)