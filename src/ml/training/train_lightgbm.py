"""
Train LightGBM model for tabular classification/regression
"""
import sys
import numpy as np
import pandas as pd
from config.databricks_config import get_spark_session
from src.ml.feature_store import FeatureStoreManager
from src.ml.models.lightgbm_model import (
    create_lightgbm_model, train_lightgbm_model,
    evaluate_classification_model, evaluate_regression_model
)
from src.ml.utils.data_preprocessing import prepare_tabular_features
from src.ml.utils.mlflow_utils import (
    setup_mlflow_experiment, log_model_parameters, log_model_metrics,
    log_lightgbm_model
)


def train_lightgbm(
    feature_table_name,
    target_col="is_env_optimal",
    task_type="classification",
    num_leaves=31,
    learning_rate=0.05,
    num_boost_round=100,
    experiment_name="hydroponics_lightgbm",
    registered_model_name="hydroponics_lightgbm_classifier"
):
    """
    Train LightGBM model for tabular data
    
    Args:
        feature_table_name: Full feature table name
        target_col: Target column name (default: "is_env_optimal" - combined optimal indicator)
        task_type: "classification" or "regression"
        num_leaves: Number of leaves
        learning_rate: Learning rate
        num_boost_round: Number of boosting rounds
        experiment_name: MLflow experiment name
        registered_model_name: Model registry name
    """
    print("="*60)
    print("TRAINING LIGHTGBM MODEL")
    print("="*60)
    
    # Setup Spark and Feature Store
    spark, config = get_spark_session()
    fs_manager = FeatureStoreManager(config.catalog)
    
    # Read features from Feature Store
    print(f"Reading features from {feature_table_name}...")
    df_features = fs_manager.get_feature_table(feature_table_name.split(".")[-1])
    
    # Convert to Pandas
    print("Converting to Pandas DataFrame...")
    df_pandas = df_features.toPandas()
    
    # Create combined environment optimal indicator if target is "is_env_optimal"
    exclude_from_features = []
    if target_col == "is_env_optimal":
        print("Creating combined environment optimal indicator...")
        optimal_cols = ["is_ph_optimal", "is_tds_optimal", "is_temp_optimal", "is_humidity_optimal"]
        
        # Check all optimal columns exist
        missing_cols = [col for col in optimal_cols if col not in df_pandas.columns]
        if missing_cols:
            raise ValueError(f"Required optimal columns not found: {missing_cols}")
        
        # Create combined target: 1 if all optimal indicators are 1, 0 otherwise
        df_pandas["is_env_optimal"] = (
            (df_pandas["is_ph_optimal"] == 1) &
            (df_pandas["is_tds_optimal"] == 1) &
            (df_pandas["is_temp_optimal"] == 1) &
            (df_pandas["is_humidity_optimal"] == 1)
        ).astype(int)
        
        # Exclude individual optimal columns from features to prevent data leakage
        exclude_from_features = optimal_cols
        print(f"Excluding optimal indicator columns from features to prevent data leakage: {exclude_from_features}")
        
        print(f"Environment optimal distribution:")
        print(df_pandas["is_env_optimal"].value_counts())
        print(f"Optimal rate: {df_pandas['is_env_optimal'].mean():.2%}")
    
    # Check target column exists
    if target_col not in df_pandas.columns:
        raise ValueError(f"Target column {target_col} not found in features")
    
    # Prepare features - exclude target and any columns used to derive it
    print("Preparing features...")
    exclude_cols = [target_col] + exclude_from_features
    feature_cols = [col for col in df_pandas.columns if col not in exclude_cols]
    X_train, X_test, y_train, y_test, feature_names = prepare_tabular_features(
        df_pandas, target_col=target_col, feature_cols=feature_cols, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {len(feature_names)}")
    
    # Create model parameters
    params, n_rounds = create_lightgbm_model(
        task_type=task_type,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        num_boost_round=num_boost_round
    )
    
    # Split training data for validation
    val_size = int(len(X_train) * 0.2)
    X_train_fit = X_train[:-val_size] if val_size > 0 else X_train
    y_train_fit = y_train[:-val_size] if val_size > 0 else y_train
    X_val = X_train[-val_size:] if val_size > 0 else None
    y_val = y_train[-val_size:] if val_size > 0 else None
    
    # Setup MLflow
    setup_mlflow_experiment(experiment_name)
    
    # Log parameters
    mlflow_params = {
        "model_type": "LightGBM",
        "task_type": task_type,
        "target_col": target_col,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "num_boost_round": num_boost_round,
        "num_features": len(feature_names),
        "train_samples": len(X_train_fit),
        "test_samples": len(X_test)
    }
    log_model_parameters(mlflow_params)
    
    # Train model
    print(f"\nTraining LightGBM model ({task_type})...")
    model = train_lightgbm_model(
        X_train=X_train_fit.values,
        y_train=y_train_fit.values,
        X_val=X_val.values if X_val is not None else None,
        y_val=y_val.values if y_val is not None else None,
        params=params,
        num_boost_round=n_rounds,
        early_stopping_rounds=10 if X_val is not None else None,
        verbose_eval=10
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    if task_type == "classification":
        test_metrics = evaluate_classification_model(model, X_test.values, y_test.values)
    else:
        test_metrics = evaluate_regression_model(model, X_test.values, y_test.values)
    
    # Verify no data leakage (optimal columns should be excluded)
    if target_col == "is_env_optimal":
        optimal_cols = ["is_ph_optimal", "is_tds_optimal", "is_temp_optimal", "is_humidity_optimal"]
        leakage_check = [col for col in optimal_cols if col in feature_names]
        if leakage_check:
            print("\n" + "!"*60)
            print("WARNING: Potential Data Leakage Detected!")
            print("!"*60)
            print(f"The following optimal indicator columns are still in the feature set:")
            for col in leakage_check:
                print(f"  - {col}")
            print("\nThese columns are used to create the target 'is_env_optimal'.")
            print("They should have been excluded from features to prevent data leakage.")
            print("!"*60 + "\n")
        else:
            print(f"\n✓ Data leakage check passed: Optimal indicator columns excluded from features.")
    
    # Log metrics
    log_model_metrics(test_metrics)
    
    print(f"\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Log model to MLflow
    print(f"\nLogging model to MLflow...")
    input_example = X_test.head(1).values if len(X_test) > 0 else None
    model_uri = log_lightgbm_model(
        model,
        artifact_path="lightgbm_model",
        registered_model_name=registered_model_name,
        input_example=input_example
    )
    print(f"Model logged: {model_uri}")
    
    # Log feature importance
    feature_importance = dict(zip(feature_names, model.feature_importance(importance_type='gain')))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Features by Importance:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.2f}")
    
    return model


if __name__ == "__main__":
    # Parameters: [DATABRICKS_CATALOG, S3_BUCKET, FEATURE_TABLE_NAME, TARGET_COL, TASK_TYPE]
    if len(sys.argv) < 3:
        raise ValueError("Usage: train_lightgbm.py <DATABRICKS_CATALOG> <S3_BUCKET> [FEATURE_TABLE_NAME] [TARGET_COL] [TASK_TYPE]")
    
    catalog = sys.argv[1]
    s3_bucket = sys.argv[2]  # Required for config
    feature_table_name = sys.argv[3]
    target_col = sys.argv[4] if len(sys.argv) > 4 else "is_env_optimal"
    
    train_lightgbm(
        feature_table_name=feature_table_name,
        target_col=target_col,
        task_type="classification",
        num_leaves=31,
        learning_rate=0.05,
        num_boost_round=100
    )
