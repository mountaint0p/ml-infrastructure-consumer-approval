# ML Infrastructure - Consumer Approval

## Description

This project aims to predict the consumer approval score for a given order based on features like order status, price, payment, etc
through a machine learning model. Furthermore,
this project creates an end-to-end machine learning workflow for both continuous training and deploying the model. It aims to implement best practices with regards to MLOps.

## Tech Stack

- Python
- Pandas
- ZenML
- MLFlow
- Scikit-learn

## Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

## Deployment Pipeline

The deployment pipeline extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. On top of the previous steps for the training pipeline, the deployment pipeline also includes the following:

- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

ZenML's deployment pipeline integrates with MLflow for tracking hyperparameters, trained models, and evaluation metrics. It logs these artifacts into a local MLflow backend. Additionally, the pipeline launches a local MLflow deployment server to serve the latest model if its accuracy surpasses a set threshold. This server operates as a daemon process, persisting beyond individual executions, and automatically updates to serve new models meeting the accuracy threshold.

## Setup

Within the Python environment of your choice, run:

```bash
pip install -r requirements.txt
```

To set up the local zenml server, run:

```bash
pip install zenml["server"]
zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## Running the pipeline

You can run two pipelines as follows:

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:

```bash
python run_deployment.py
```

## Demo Streamlit App

To demo the model, run the command

```bash
streamlit run streamlit_app.py
```
