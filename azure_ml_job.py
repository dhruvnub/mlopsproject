# azure_ml_job.py — Experiment 3
# Called by Jenkins and GitHub Actions to submit training job to Azure ML

import argparse
import time
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import ClientSecretCredential


def submit(args):
    print("\n Authenticating with Azure...")
    cred = ClientSecretCredential(
        tenant_id     = args.tenant_id,
        client_id     = args.client_id,
        client_secret = args.client_secret,
    )
    ml = MLClient(
        credential          = cred,
        subscription_id     = args.subscription_id,
        resource_group_name = args.resource_group,
        workspace_name      = args.workspace,
    )
    print(f"Connected to workspace: {args.workspace}")

    # Custom environment with all required packages
    env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file={
            "name": "placement-env",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {"pip": [
                    "pandas",
                    "scikit-learn",
                    "mlflow",
                    "azureml-mlflow",
                    "joblib",
                    "python-dotenv"
                ]}
            ]
        },
        name="placement-training-env",
    )

    job = command(
        display_name    = f"jenkins-placement-{int(time.time())}",
        command         = "python train.py",
        code            = ".",
        environment     = env,
        compute         = args.compute,
        experiment_name = args.experiment,
        environment_variables={
            "MLFLOW_TRACKING_URI": "azureml://tracking"
        },
    )

    submitted = ml.jobs.create_or_update(job)
    print(f"\n Job submitted!")
    print(f"   Name   : {submitted.name}")
    print(f"   Status : {submitted.status}")
    print(f"   View   : https://ml.azure.com")

    print("\n Waiting for job to complete...")
    while True:
        status = ml.jobs.get(submitted.name).status
        print(f"   Status: {status}")
        if status in ("Completed", "Finished"):
            print("\n Job completed successfully!")
            break
        elif status in ("Failed", "Canceled"):
            print("\n Job failed. Check Azure ML Studio → Jobs → Logs")
            raise RuntimeError(f"Azure ML job {status}")
        time.sleep(30)

    return submitted.name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--client-id",       required=True)
    p.add_argument("--client-secret",   required=True)
    p.add_argument("--tenant-id",       required=True)
    p.add_argument("--subscription-id", required=True)
    p.add_argument("--resource-group",  required=True)
    p.add_argument("--workspace",       required=True)
    p.add_argument("--experiment",      default="placement-prediction")
    p.add_argument("--compute",         default="placement-cluster")
    submit(p.parse_args())
