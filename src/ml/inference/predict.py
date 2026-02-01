"""
Model inference script - load models from MLflow registry and make predictions
"""
import sys
import numpy as np
import pandas as pd
import torch
from config.databricks_config import get_spark_session
from ml.feature_store import FeatureStoreManager
from ml.utils.mlflow_utils import load_model_from_registry
from ml.utils.data_preprocessing import create_sequences, scale_features


def predict_with_lstm_gru(
    model_name,
    feature_table_name,
    target_col="ph_level",
    sequence_length=24,
    model_stage="Production",
    num_predictions=10
):
    """
    Make predictions using LSTM or GRU model from MLflow registry
    
    Args:
        model_name: Model name in MLflow registry
        feature_table_name: Feature table name
        target_col: Target column (for feature selection)
        sequence_length: Sequence length used during training
        model_stage: Model stage (Production, Staging, etc.)
        num_predictions: Number of predictions to make
    """
    print("="*60)
    print(f"LSTM/GRU PREDICTION - {model_name}")
    print("="*60)
    
    # Load model from registry
    print(f"Loading model {model_name} from {model_stage} stage...")
    model = load_model_from_registry(model_name, stage=model_stage)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Load features
    spark, config = get_spark_session()
    fs_manager = FeatureStoreManager(config.catalog)
    
    print(f"Reading features from {feature_table_name}...")
    df_features = fs_manager.get_feature_table(feature_table_name.split(".")[-1])
    df_pandas = df_features.orderBy("timestamp").toPandas()
    
    # Select numeric columns
    numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns.tolist()
    
    # Get last sequence_length rows for prediction
    if len(df_pandas) < sequence_length:
        raise ValueError(f"Not enough data. Need at least {sequence_length} rows, got {len(df_pandas)}")
    
    # Create sequence
    last_sequence = df_pandas[numeric_cols].tail(sequence_length).values
    last_sequence = last_sequence.reshape(1, sequence_length, -1)
    
    # Scale (in production, you'd load the scaler from MLflow artifacts)
    # For now, we'll use a simple standardization
    last_sequence_scaled = (last_sequence - last_sequence.mean()) / (last_sequence.std() + 1e-8)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(last_sequence_scaled).to(device)
    
    # Make predictions
    print(f"\nMaking {num_predictions} predictions...")
    predictions = []
    
    with torch.no_grad():
        current_sequence = X_tensor
        
        for i in range(num_predictions):
            pred = model(current_sequence)
            predictions.append(pred.cpu().numpy()[0, 0])
            
            # Update sequence (rolling window)
            # In practice, you'd use actual new data
            # For demo, we'll shift and append prediction
            new_seq = torch.cat([current_sequence[:, 1:, :], pred.unsqueeze(1)], dim=1)
            current_sequence = new_seq
    
    print(f"\nPredictions for {target_col}:")
    for i, pred in enumerate(predictions, 1):
        print(f"  Step {i}: {pred:.4f}")
    
    return predictions


def predict_with_lightgbm(
    model_name,
    feature_table_name,
    target_col="is_ph_optimal",
    model_stage="Production",
    num_predictions=10
):
    """
    Make predictions using LightGBM model from MLflow registry
    
    Args:
        model_name: Model name in MLflow registry
        feature_table_name: Feature table name
        target_col: Target column name
        model_stage: Model stage (Production, Staging, etc.)
        num_predictions: Number of predictions to make
    """
    print("="*60)
    print(f"LIGHTGBM PREDICTION - {model_name}")
    print("="*60)
    
    # Load model from registry
    print(f"Loading model {model_name} from {model_stage} stage...")
    model = load_model_from_registry(model_name, stage=model_stage)
    print("Model loaded")
    
    # Load features
    spark, config = get_spark_session()
    fs_manager = FeatureStoreManager(config.catalog)
    
    print(f"Reading features from {feature_table_name}...")
    df_features = fs_manager.get_feature_table(feature_table_name.split(".")[-1])
    df_pandas = df_features.toPandas()
    
    # Select features (exclude target)
    feature_cols = [col for col in df_pandas.columns if col != target_col]
    X = df_pandas[feature_cols].select_dtypes(include=[np.number])
    
    # Get last num_predictions rows
    X_pred = X.tail(num_predictions)
    
    # Make predictions
    print(f"\nMaking predictions on {len(X_pred)} samples...")
    predictions = model.predict(X_pred.values)
    
    # For classification, convert probabilities to classes
    if len(predictions.shape) == 1 and predictions.max() <= 1.0:
        pred_classes = (predictions > 0.5).astype(int)
        print(f"\nPredictions for {target_col}:")
        for i, (prob, cls) in enumerate(zip(predictions, pred_classes), 1):
            print(f"  Sample {i}: Probability={prob:.4f}, Class={cls}")
    else:
        print(f"\nPredictions for {target_col}:")
        for i, pred in enumerate(predictions, 1):
            print(f"  Sample {i}: {pred:.4f}")
    
    return predictions


if __name__ == "__main__":
    # Parameters: [DATABRICKS_CATALOG, S3_BUCKET, MODEL_NAME, MODEL_TYPE, FEATURE_TABLE_NAME]
    if len(sys.argv) < 5:
        raise ValueError("Usage: predict.py <DATABRICKS_CATALOG> <S3_BUCKET> <MODEL_NAME> <MODEL_TYPE> [FEATURE_TABLE_NAME] [MODEL_STAGE]")
    
    catalog = sys.argv[1]
    s3_bucket = sys.argv[2]
    model_name = sys.argv[3]
    model_type = sys.argv[4].lower()  # "lstm", "gru", or "lightgbm"
    feature_table_name = sys.argv[5] if len(sys.argv) > 5 else f"{catalog}.feature_store.sensor_features"
    model_stage = sys.argv[6] if len(sys.argv) > 6 else "Production"
    
    if model_type in ["lstm", "gru"]:
        predict_with_lstm_gru(
            model_name=model_name,
            feature_table_name=feature_table_name,
            model_stage=model_stage
        )
    elif model_type == "lightgbm":
        predict_with_lightgbm(
            model_name=model_name,
            feature_table_name=feature_table_name,
            model_stage=model_stage
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm', 'gru', or 'lightgbm'")
