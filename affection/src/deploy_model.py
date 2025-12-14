import argparse
import os
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration
from azure.ai.ml.entities import OnlineEndpoint

def main(model_id_arg: str, endpoint_name: str, deployment_name: str):
    if os.path.isfile(model_id_arg):
        with open(model_id_arg, "r") as f:
            model_id = f.read().strip()
    else:
        model_id = model_id_arg
    try:
        ml_client = MLClient.from_config(DefaultAzureCredential())
    except Exception:
        ml_client = MLClient.from_config(InteractiveBrowserCredential())
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_id,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py"
        ),
        environment="azureml:TCBA_ML_A_3:4",
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
    print(f"[DEPLOYED] Model {model_id} ke endpoint {endpoint_name} dengan deployment {deployment_name}")
    
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"[TRAFFIC] Endpoint {endpoint_name} diarahkan 100% ke {deployment_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="File path berisi model ID")
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, required=True)
    args = parser.parse_args()

    main(args.model_id, args.endpoint_name, args.deployment_name)