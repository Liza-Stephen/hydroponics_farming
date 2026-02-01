"""
MLflow utilities for experiment tracking and model registry
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from datetime import datetime


def _get_experiment_path(experiment_name):
    """
    Convert experiment name to Databricks absolute path format
    
    Args:
        experiment_name: Simple experiment name (e.g., 'hydroponics_lstm')
    
    Returns:
        Absolute path (e.g., '/Users/<username>/hydroponics_lstm')
    """
    # If already an absolute path, return as-is
    if experiment_name.startswith("/"):
        return experiment_name
    
    # Try to get username from Databricks context
    try:
        # Method 1: Try dbutils (works in notebooks)
        try:
            from pyspark.dbutils import DBUtils
            dbutils = DBUtils()
            username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
            if username:
                return f"/Users/{username}/{experiment_name}"
        except:
            pass
        
        # Method 2: Try Spark context (works in jobs)
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                # Try to get user from Spark context
                username = spark.sparkContext.getConf().get("spark.databricks.clusterUsageTags.userEmail", "")
                if username:
                    # Extract username from email if needed
                    username = username.split("@")[0] if "@" in username else username
                    return f"/Users/{username}/{experiment_name}"
        except:
            pass
        
        # Method 3: Try environment variable
        import os
        username = os.environ.get("USER", os.environ.get("USERNAME", ""))
        if username:
            return f"/Users/{username}/{experiment_name}"
    except:
        pass
    
    # Fallback: return as-is and let MLflow handle it (might work in some cases)
    # Or use a shared location
    return f"/Shared/{experiment_name}"


def setup_mlflow_experiment(experiment_name):
    """
    Setup MLflow experiment
    
    Args:
        experiment_name: Name of the experiment (will be converted to absolute path if needed)
    
    Returns:
        experiment_id
    """
    # Convert to absolute path format for Databricks
    experiment_path = _get_experiment_path(experiment_name)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_path)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_path)
            print(f"Created MLflow experiment: {experiment_path}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing MLflow experiment: {experiment_path}")
        
        mlflow.set_experiment(experiment_path)
        return experiment_id
    except Exception as e:
        print(f"Error setting up experiment '{experiment_path}': {e}")
        # Fallback: try to use a default experiment in Shared location
        try:
            default_path = _get_experiment_path("default")
            experiment = mlflow.get_experiment_by_name(default_path)
            if experiment is None:
                experiment_id = mlflow.create_experiment(default_path)
                print(f"Created fallback MLflow experiment: {default_path}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using fallback MLflow experiment: {default_path}")
            mlflow.set_experiment(default_path)
            return experiment_id
        except Exception as fallback_error:
            print(f"Warning: Could not set up MLflow experiment (fallback also failed: {fallback_error})")
            print("Continuing without experiment tracking - model will still be trained but not logged to MLflow")
            # Return None to indicate experiment setup failed
            return None


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
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=artifact_path
    )
    
    # Extract model URI from ModelInfo object (newer MLflow versions return ModelInfo)
    if hasattr(model_info, 'model_uri'):
        model_uri = model_info.model_uri
    elif isinstance(model_info, str):
        model_uri = model_info
    else:
        # Fallback: construct URI from run_id and artifact_path
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
    
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
    model_info = mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path=artifact_path,
        input_example=input_example
    )
    
    # Extract model URI from ModelInfo object (newer MLflow versions return ModelInfo)
    if hasattr(model_info, 'model_uri'):
        model_uri = model_info.model_uri
    elif isinstance(model_info, str):
        model_uri = model_info
    else:
        # Fallback: construct URI from run_id and artifact_path
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
    
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
