"""
MLflow utilities for experiment tracking and model registry
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from datetime import datetime


def setup_mlflow_experiment(experiment_name):
    """
    Setup MLflow experiment
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        experiment_id
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created MLflow experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        print(f"Error setting up experiment: {e}")
        # Fallback to default experiment
        mlflow.set_experiment("Default")
        return "0"


def log_model_parameters(params):
    """Log model parameters to MLflow"""
    mlflow.log_params(params)


def log_model_metrics(metrics, step=None):
    """Log model metrics to MLflow"""
    if step is not None:
        mlflow.log_metrics(metrics, step=step)
    else:
        mlflow.log_metrics(metrics)


def log_pytorch_model(model, artifact_path="model", registered_model_name=None):
    """
    Log PyTorch model to MLflow
    
    Args:
        model: PyTorch model
        artifact_path: Path to save the model artifact
        registered_model_name: Name for model registry (optional)
    
    Returns:
        model_uri
    """
    model_uri = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=artifact_path
    )
    
    if registered_model_name:
        mlflow.register_model(model_uri, registered_model_name)
        print(f"Registered model: {registered_model_name}")
    
    return model_uri


def log_lightgbm_model(model, artifact_path="model", registered_model_name=None, input_example=None):
    """
    Log LightGBM model to MLflow
    
    Args:
        model: LightGBM model
        artifact_path: Path to save the model artifact
        registered_model_name: Name for model registry (optional)
        input_example: Example input for model signature (optional)
    
    Returns:
        model_uri
    """
    model_uri = mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path=artifact_path,
        input_example=input_example
    )
    
    if registered_model_name:
        mlflow.register_model(model_uri, registered_model_name)
        print(f"Registered model: {registered_model_name}")
    
    return model_uri


def get_latest_model_version(model_name):
    """Get latest model version from MLflow model registry"""
    client = MlflowClient()
    try:
        latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        if latest_version:
            return latest_version[0].version
        return None
    except Exception as e:
        print(f"Error getting latest model version: {e}")
        return None


def transition_model_stage(model_name, version, stage):
    """
    Transition model to a new stage (Staging, Production, Archived)
    
    Args:
        model_name: Name of the model in registry
        version: Model version
        stage: Target stage (Staging, Production, Archived)
    """
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Transitioned {model_name} v{version} to {stage}")
    except Exception as e:
        print(f"Error transitioning model: {e}")


def load_model_from_registry(model_name, stage="Production"):
    """
    Load model from MLflow model registry
    
    Args:
        model_name: Name of the model
        stage: Model stage (Production, Staging, etc.)
    
    Returns:
        Loaded model
    """
    model_uri = f"models:/{model_name}/{stage}"
    try:
        # Try PyTorch first
        model = mlflow.pytorch.load_model(model_uri)
        return model
    except:
        try:
            # Try LightGBM
            model = mlflow.lightgbm.load_model(model_uri)
            return model
        except Exception as e:
            raise ValueError(f"Could not load model {model_name} from registry: {e}")
