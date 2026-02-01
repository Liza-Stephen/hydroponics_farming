"""
Train GRU model for time-series forecasting
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from config.databricks_config import get_spark_session
from src.ml.feature_store import FeatureStoreManager
from src.ml.models.gru_model import create_gru_model
from src.ml.utils.data_preprocessing import create_sequences, scale_features, calculate_time_series_metrics
from src.ml.utils.mlflow_utils import (
    setup_mlflow_experiment, log_model_parameters, log_model_metrics,
    log_pytorch_model
)


def train_gru(
    feature_table_name,
    target_col="ph_level",
    sequence_length=24,
    forecast_horizon=1,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    experiment_name="hydroponics_gru",
    registered_model_name="hydroponics_gru_forecast"
):
    """
    Train GRU model for time-series forecasting
    
    Args:
        feature_table_name: Full feature table name
        target_col: Column to predict
        sequence_length: Number of time steps for input sequence
        forecast_horizon: Number of steps ahead to predict
        hidden_size: GRU hidden size
        num_layers: Number of GRU layers
        dropout: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of training epochs
        experiment_name: MLflow experiment name
        registered_model_name: Model registry name
    """
    print("="*60)
    print("TRAINING GRU MODEL")
    print("="*60)
    
    # Setup Spark and Feature Store
    spark, config = get_spark_session()
    fs_manager = FeatureStoreManager(config.catalog)
    
    # Read features from Feature Store
    print(f"Reading features from {feature_table_name}...")
    df_features = fs_manager.get_feature_table(feature_table_name.split(".")[-1])
    
    # Convert to Pandas (for time-series processing)
    print("Converting to Pandas DataFrame...")
    df_pandas = df_features.orderBy("timestamp").toPandas()
    
    # Select relevant columns
    numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"Target column {target_col} not found in features")
    
    # Create sequences
    print(f"Creating sequences (length={sequence_length}, horizon={forecast_horizon})...")
    X, y = create_sequences(
        df_pandas[numeric_cols],
        sequence_length=sequence_length,
        target_col=target_col,
        forecast_horizon=forecast_horizon
    )
    
    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    # Split data (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler_X = scale_features(X_train, X_test, scaler_type="standard")
    
    # Scale targets
    y_train_2d = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train.reshape(-1, y_train.shape[-1])
    y_test_2d = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test.reshape(-1, y_test.shape[-1])
    y_train_scaled, y_test_scaled, scaler_y = scale_features(
        y_train_2d, y_test_2d, scaler_type="standard"
    )
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    
    # Create model
    input_size = X_train_scaled.shape[2]
    output_size = y_train_scaled.shape[1] if len(y_train_scaled.shape) > 1 else 1
    
    model, optimizer, criterion = create_gru_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        learning_rate=learning_rate
    )
    model = model.to(device)
    
    # Setup MLflow
    setup_mlflow_experiment(experiment_name)
    
    # Log parameters
    params = {
        "model_type": "GRU",
        "target_col": target_col,
        "sequence_length": sequence_length,
        "forecast_horizon": forecast_horizon,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "input_size": input_size,
        "output_size": output_size
    }
    log_model_parameters(params)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        # Mini-batch training
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
            
            # Calculate metrics
            y_pred_scaled = val_outputs.cpu().numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_test_scaled)
            
            metrics = calculate_time_series_metrics(y_true.flatten(), y_pred.flatten())
            metrics["train_loss"] = avg_loss
            metrics["val_loss"] = val_loss
            
            log_model_metrics(metrics, step=epoch + 1)
            
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val MAE: {metrics['mae']:.4f}, Val RMSE: {metrics['rmse']:.4f}")
            
            model.train()
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    final_metrics = calculate_time_series_metrics(y_true.flatten(), y_pred.flatten())
    log_model_metrics(final_metrics, step=epochs)
    
    print(f"\nFinal Test Metrics:")
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  RMSE: {final_metrics['rmse']:.4f}")
    if final_metrics['mape']:
        print(f"  MAPE: {final_metrics['mape']:.2f}%")
    
    # Log model to MLflow
    print(f"\nLogging model to MLflow...")
    model_uri = log_pytorch_model(model, artifact_path="gru_model", registered_model_name=registered_model_name)
    print(f"Model logged: {model_uri}")
    
    return model, scaler_X, scaler_y


if __name__ == "__main__":
    # Parameters: [DATABRICKS_CATALOG, S3_BUCKET, FEATURE_TABLE_NAME, TARGET_COL]
    if len(sys.argv) < 3:
        raise ValueError("Usage: train_gru.py <DATABRICKS_CATALOG> <S3_BUCKET> [FEATURE_TABLE_NAME] [TARGET_COL]")
    
    catalog = sys.argv[1]
    s3_bucket = sys.argv[2]  # Required for config
    feature_table_name = sys.argv[3] if len(sys.argv) > 3 else f"{catalog}.feature_store.sensor_features"
    target_col = sys.argv[4] if len(sys.argv) > 4 else "ph_level"
    
    train_gru(
        feature_table_name=feature_table_name,
        target_col=target_col,
        sequence_length=24,
        forecast_horizon=1,
        hidden_size=64,
        num_layers=2,
        epochs=50
    )
