// Jenkinsfile — Experiment 3: Jenkins → Azure ML Training

pipeline {
    agent any

    environment {
        PYTHON         = '"C:\\Users\\inYodreamzz\\Desktop\\mlops-project\\mlops_env\\Scripts\\python.exe"'
        RESOURCE_GROUP = 'rg-mlops-exp'
        WORKSPACE_NAME = 'mlops-workspace'
        EXPERIMENT     = 'placement-prediction'
        COMPUTE        = 'placement-cluster'
        ACR_NAME       = 'placementacr.azurecr.io'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python packages...'
                bat "%PYTHON% -m pip install -r requirements.txt"
            }
        }

        stage('Pull Data from Azure Blob via DVC') {
            steps {
                echo 'Pulling data and model from Azure Blob Storage...'
                withCredentials([
                    string(credentialsId: 'AZURE_STORAGE_CONNECTION_STRING', variable: 'AZURE_STORAGE_CONNECTION_STRING'),
                ]) {
                    bat "dvc pull --force"
                }
            }
        }

        stage('Train Model') {
            steps {
                echo 'Training placement prediction model...'
                bat """
                    set MLFLOW_TRACKING_URI=mlruns
                    %PYTHON% train.py
                """
            }
        }

        stage('Submit Azure ML Job') {
            steps {
                echo 'Submitting training job to Azure ML...'
                withCredentials([
                    string(credentialsId: 'AZURE_CLIENT_ID',       variable: 'AZ_CLIENT_ID'),
                    string(credentialsId: 'AZURE_CLIENT_SECRET',    variable: 'AZ_CLIENT_SECRET'),
                    string(credentialsId: 'AZURE_TENANT_ID',        variable: 'AZ_TENANT_ID'),
                    string(credentialsId: 'AZURE_SUBSCRIPTION_ID',  variable: 'AZ_SUB_ID'),
                ]) {
                    bat """
                        %PYTHON% azure_ml_job.py ^
                            --client-id       %AZ_CLIENT_ID%     ^
                            --client-secret   %AZ_CLIENT_SECRET% ^
                            --tenant-id       %AZ_TENANT_ID%     ^
                            --subscription-id %AZ_SUB_ID%        ^
                            --resource-group  %RESOURCE_GROUP%   ^
                            --workspace       %WORKSPACE_NAME%   ^
                            --experiment      %EXPERIMENT%       ^
                            --compute         %COMPUTE%
                    """
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                bat 'docker build -t placement-api .'
            }
        }

        stage('Push to Azure Container Registry') {
            steps {
                echo 'Pushing image to Azure Container Registry...'
                withCredentials([
                    string(credentialsId: 'AZURE_CLIENT_ID',     variable: 'AZ_CLIENT_ID'),
                    string(credentialsId: 'AZURE_CLIENT_SECRET', variable: 'AZ_CLIENT_SECRET'),
                ]) {
                    bat """
                        docker login %ACR_NAME% -u %AZ_CLIENT_ID% -p %AZ_CLIENT_SECRET%
                        docker tag placement-api %ACR_NAME%/placement-api:latest
                        docker push %ACR_NAME%/placement-api:latest
                    """
                }
            }
        }

    }

    post {
        success {
            echo 'Pipeline complete! Model trained, Azure ML job submitted, image pushed to ACR.'
            echo 'View results: https://ml.azure.com'
        }
        failure {
            echo 'Pipeline failed. Check logs above.'
        }
    }
}
