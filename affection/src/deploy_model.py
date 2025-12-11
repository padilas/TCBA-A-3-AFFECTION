import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration

def main(model_id_file: str, endpoint_name: str, deployment_name: str):
    with open(model_id_file, "r") as f:
        model_id = f.read().strip()
    ml_client = MLClient.from_config(DefaultAzureCredential())
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_id,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py"
        ),
        environment="azureml:TCBA-A-3:1",
        instance_type="Standard_F4s_v2",
        instance_count=1,
        environment_variables={
            "sql_server": os.environ.get("sql_server", ""),
            "sql_database": os.environ.get("sql_database", ""),
            "sql_username": os.environ.get("sql_username", ""),
            "sql_password": os.environ.get("sql_password", ""),
            "sql_table": os.environ.get("sql_table", ""),
        }
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"[DEPLOYED] Model {model_id} ke endpoint {endpoint_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="File path berisi model ID")
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, required=True)
    args = parser.parse_args()

    main(args.model_id, args.endpoint_name, args.deployment_name)
