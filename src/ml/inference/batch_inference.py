"""
Batch inference script - load latest model versions and perform batch predictions
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow.tracking import MlflowClient
from config.databricks_config import get_spark_session
from src.ml.feature_store import FeatureStoreManager
from src.ml.utils.mlflow_utils import load_model_from_registry, get_latest_model_version
# Note: We create sequences manually in this script


def load_latest_model_and_uri(model_name, catalog=None):
    """
    Load the latest model version and return model + URI
    
    Args:
        model_name: Model name in MLflow registry (can be simple or full three-level format)
        catalog: Catalog name (optional, for constructing full path)
    
    Returns:
        (model, model_uri) tuple
    """
    # Get latest version (function handles both simple and full names)
    version, full_model_name = get_latest_model_version(model_name, catalog=catalog)
    if version is None:
        raise ValueError(f"No versions found for model {model_name}")
    
    model_uri = f"models:/{full_model_name}/{version}"
    print(f"Loading {full_model_name} version {version} from {model_uri}")
    
    # Load model
    try:
        model = mlflow.pytorch.load_model(model_uri)
        if hasattr(model, 'eval'):
            model.eval()
        return model, model_uri
    except:
        try:
            model = mlflow.lightgbm.load_model(model_uri)
            return model, model_uri
        except Exception as e:
            raise ValueError(f"Could not load model {model_name} from registry: {e}")


def load_feature_spec(model_uri, feature_table_name, target_col=None):
    """
    Load feature specification from model or infer from feature table
    
    Args:
        model_uri: MLflow model URI
        feature_table_name: Feature table name
        target_col: Target column name (to exclude)
    
    Returns:
        List of feature column names
    """
    spark, config = get_spark_session()
    fs_manager = FeatureStoreManager(config.catalog)
    
    # Try to get feature names from model signature
    try:
        model_info = mlflow.models.get_model_info(model_uri)
        if hasattr(model_info, 'signature') and model_info.signature is not None:
            if hasattr(model_info.signature, 'inputs') and model_info.signature.inputs is not None:
                # Extract feature names from signature
                feature_names = [input.name for input in model_info.signature.inputs.inputs]
                if feature_names:
                    # Check if feature names are integers (indicates unnamed columns from array)
                    # If so, fall back to inferring from feature table
                    try:
                        # Try to convert first name to int - if it works, they're all likely integers
                        int(feature_names[0])
                        print(f"Model signature has integer column names, inferring from feature table...")
                        # Fall through to feature table inference
                    except (ValueError, TypeError):
                        # Feature names are actual strings, use them
                        print(f"Loaded {len(feature_names)} features from model signature")
                        return feature_names
    except Exception as e:
        print(f"Could not load features from signature: {e}")
    
    # Fallback: infer from feature table
    print(f"Inferring features from feature table {feature_table_name}...")
    df_features = fs_manager.get_feature_table(feature_table_name.split(".")[-1])
    df_pandas = df_features.toPandas()
    
    # Get numeric columns, exclude timestamp and target
    exclude_cols = ['timestamp', 'timestamp_key']
    if target_col:
        exclude_cols.append(target_col)
    
    # If is_env_optimal exists in the table, exclude optimal indicators and raw sensor values
    # (these would cause data leakage if used as features for is_env_optimal prediction)
    if "is_env_optimal" in df_pandas.columns:
        optimal_cols = ["is_ph_optimal", "is_tds_optimal", "is_temp_optimal", "is_humidity_optimal"]
        raw_sensor_cols = ["ph_level", "tds_level", "air_temperature", "air_humidity"]
        # Only exclude if they exist in the table
        exclude_cols.extend([col for col in optimal_cols if col in df_pandas.columns])
        exclude_cols.extend([col for col in raw_sensor_cols if col in df_pandas.columns])
    
    feature_cols = [
        col for col in df_pandas.select_dtypes(include=[np.number]).columns 
        if col not in exclude_cols
    ]
    
    print(f"Inferred {len(feature_cols)} features from feature table")
    return feature_cols


def batch_inference(
    catalog,
    s3_bucket,
    feature_table_name,
    lstm_model_name,
    gru_model_name,
    lightgbm_model_name,
    output_table_name,
    sequence_length=24,
    target_col="ph_level"
):
    """
    Perform batch inference using latest model versions
    
    Args:
        catalog: Unity Catalog name
        s3_bucket: S3 bucket name
        feature_table_name: Feature table name (e.g., "hydroponics.feature_store.sensor_features")
        lstm_model_name: LSTM model name in registry
        gru_model_name: GRU model name in registry
        lightgbm_model_name: LightGBM model name in registry
        output_table_name: Output table name for predictions (e.g., "hydroponics.predictions.batch_predictions")
        sequence_length: Sequence length for LSTM/GRU models
        target_col: Target column name
    """
    print("\nStarting batch inference...\n")
    
    # 1. Read Feature Store (ORDER MATTERS)
    spark, config = get_spark_session()
    spark_df = spark.table(f"{catalog}.{feature_table_name}").orderBy("timestamp")
    pdf = spark_df.toPandas()
    
    print(f"Loaded {len(pdf)} feature rows")
    
    # 2. Load models + feature specs
    print("\nLoading models...")
    lstm_model, lstm_uri = load_latest_model_and_uri(lstm_model_name, catalog=catalog)
    gru_model, gru_uri = load_latest_model_and_uri(gru_model_name, catalog=catalog)
    lgbm_model, lgbm_uri = load_latest_model_and_uri(lightgbm_model_name, catalog=catalog)
    
    print("\nLoading feature specifications...")
    lstm_features = load_feature_spec(lstm_uri, feature_table_name, target_col)
    gru_features = load_feature_spec(gru_uri, feature_table_name, target_col)
    lgbm_features = load_feature_spec(lgbm_uri, feature_table_name, target_col)
    
    # Sanity check
    assert lstm_features == gru_features, "LSTM and GRU feature specs differ!"
    print(f"LSTM/GRU features match: {len(lstm_features)} features")
    print(f"LightGBM features: {len(lgbm_features)} features")
    
    # 3. ----- LightGBM inference (tabular) -----
    print("\nRunning LightGBM inference...")
    X_tabular = pdf[lgbm_features].fillna(0).astype("float32")
    pdf["lightgbm_prediction"] = lgbm_model.predict(X_tabular.values)
    print(f"LightGBM predictions: {len(pdf['lightgbm_prediction'])} samples")
    
    # 4. ----- LSTM / GRU inference (sequence) -----
    print("\nRunning LSTM/GRU inference...")
    X_raw = pdf[lstm_features].fillna(0).values.astype("float32")
    
    # Create sequences manually (since create_sequences expects DataFrame)
    # We'll create sequences from the numpy array
    X_seq = []
    for i in range(len(X_raw) - sequence_length + 1):
        X_seq.append(X_raw[i:i + sequence_length])
    X_seq = np.array(X_seq)
    print(f"Created {len(X_seq)} sequences from {len(pdf)} rows")
    
    # Prepare models for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = lstm_model.to(device)
    gru_model = gru_model.to(device)
    
    # Convert to tensors and scale (simple standardization)
    X_seq_tensor = torch.FloatTensor(X_seq).to(device)
    # Simple scaling: normalize each feature
    X_seq_mean = X_seq_tensor.mean(dim=(0, 1), keepdim=True)
    X_seq_std = X_seq_tensor.std(dim=(0, 1), keepdim=True) + 1e-8
    X_seq_scaled = (X_seq_tensor - X_seq_mean) / X_seq_std
    
    # Make predictions
    with torch.no_grad():
        lstm_preds = lstm_model(X_seq_scaled).cpu().numpy().flatten()
        gru_preds = gru_model(X_seq_scaled).cpu().numpy().flatten()
    
    print(f"LSTM predictions: {len(lstm_preds)} samples")
    print(f"GRU predictions: {len(gru_preds)} samples")
    
    # Align predictions with timestamps
    # Sequences start at index sequence_length, so we need to align
    seq_df = pdf.iloc[sequence_length:].copy()
    seq_df["lstm_prediction"] = lstm_preds
    seq_df["gru_prediction"] = gru_preds
    seq_df["lightgbm_prediction"] = pdf.iloc[sequence_length:]["lightgbm_prediction"].values
    
    # 5. Ensemble (optional)
    seq_df["ensemble_avg"] = (
        seq_df["lstm_prediction"]
        + seq_df["gru_prediction"]
        + seq_df["lightgbm_prediction"]
    ) / 3
    
    # 6. Metadata
    seq_df["inference_timestamp"] = pd.Timestamp.utcnow()
    
    # 7. Write predictions
    print(f"\nWriting predictions to {output_table_name}...")
    
    # Create database if it doesn't exist
    db_parts = output_table_name.split(".")
    if len(db_parts) == 3:
        catalog_name, db_name, table_name = db_parts
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog_name}.{db_name}")
    elif len(db_parts) == 2:
        db_name, table_name = db_parts
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.{db_name}")
        output_table_name = f"{catalog}.{output_table_name}"
    else:
        table_name = output_table_name
        output_table_name = f"{catalog}.predictions.{table_name}"
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.predictions")
    
    spark.createDataFrame(seq_df).write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(output_table_name)
    
    print(f"\nInference complete.")
    print(f"Predictions written to {output_table_name}")
    print(f"Total predictions: {len(seq_df)}")
    print(f"\nPrediction statistics:")
    print(f"  LSTM - Mean: {seq_df['lstm_prediction'].mean():.4f}, Std: {seq_df['lstm_prediction'].std():.4f}")
    print(f"  GRU - Mean: {seq_df['gru_prediction'].mean():.4f}, Std: {seq_df['gru_prediction'].std():.4f}")
    print(f"  LightGBM - Mean: {seq_df['lightgbm_prediction'].mean():.4f}, Std: {seq_df['lightgbm_prediction'].std():.4f}")
    print(f"  Ensemble - Mean: {seq_df['ensemble_avg'].mean():.4f}, Std: {seq_df['ensemble_avg'].std():.4f}")


if __name__ == "__main__":
    # Parameters: [DATABRICKS_CATALOG, S3_BUCKET, FEATURE_TABLE_NAME, LSTM_MODEL_NAME, GRU_MODEL_NAME, LIGHTGBM_MODEL_NAME, OUTPUT_TABLE_NAME, SEQUENCE_LENGTH, TARGET_COL]
    if len(sys.argv) < 7:
        raise ValueError(
            "Usage: batch_inference.py <DATABRICKS_CATALOG> <S3_BUCKET> <FEATURE_TABLE_NAME> "
            "<LSTM_MODEL_NAME> <GRU_MODEL_NAME> <LIGHTGBM_MODEL_NAME> <OUTPUT_TABLE_NAME> "
            "[SEQUENCE_LENGTH] [TARGET_COL]"
        )
    
    catalog = sys.argv[1]
    s3_bucket = sys.argv[2]
    feature_table_name = sys.argv[3]
    lstm_model_name = sys.argv[4]
    gru_model_name = sys.argv[5]
    lightgbm_model_name = sys.argv[6]
    output_table_name = sys.argv[7]
    sequence_length = int(sys.argv[8]) if len(sys.argv) > 8 else 24
    target_col = sys.argv[9] if len(sys.argv) > 9 else "ph_level"
    
    batch_inference(
        catalog=catalog,
        s3_bucket=s3_bucket,
        feature_table_name=feature_table_name,
        lstm_model_name=lstm_model_name,
        gru_model_name=gru_model_name,
        lightgbm_model_name=lightgbm_model_name,
        output_table_name=output_table_name,
        sequence_length=sequence_length,
        target_col=target_col
    )
