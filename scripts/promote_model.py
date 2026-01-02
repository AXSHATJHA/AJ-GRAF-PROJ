# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "AXSHATJHA"
    repo_name = "AJ-GRAF-PROJ"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "my_model"

    # 1. Get the version currently aliased as "staging"
    # (Assumes you have assigned the alias "staging" to your candidate model)
    staging_model = client.get_model_version_by_alias(model_name, "staging")
    latest_version_staging = staging_model.version

    # 2. Promote to Production (Assign "prod" alias)
    # NOTE: This replaces the "Archive" loop. Assigning the alias to the new version 
    # automatically removes it from the old version.
    client.set_registered_model_alias(
        name=model_name,
        alias="prod", 
        version=latest_version_staging
    )

    print(f"Model version {latest_version_staging} promoted to Production (alias 'prod')")

if __name__ == "__main__":
    promote_model()