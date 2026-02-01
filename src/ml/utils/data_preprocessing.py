"""
Data preprocessing utilities for time-series and tabular models
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def create_sequences(data, sequence_length, target_col=None, forecast_horizon=1):
    """
    Create sequences for time-series models (LSTM/GRU)
    
    Args:
        data: DataFrame with time-series data
        sequence_length: Number of time steps to use as input
        target_col: Column name to predict (if None, uses all numeric columns)
        forecast_horizon: Number of steps ahead to predict (default: 1)
    
    Returns:
        X: Input sequences (samples, sequence_length, features)
        y: Target values (samples, forecast_horizon, features) or (samples, forecast_horizon)
    """
    if target_col:
        # Single target prediction
        target_data = data[[target_col]].values
        feature_data = data.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    else:
        # Multi-target prediction (all numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        target_data = data[numeric_cols].values
        feature_data = target_data
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(feature_data[i:i + sequence_length])
        if forecast_horizon == 1:
            y.append(target_data[i + sequence_length])
        else:
            y.append(target_data[i + sequence_length:i + sequence_length + forecast_horizon])
    
    return np.array(X), np.array(y)


def prepare_tabular_features(df, target_col, feature_cols=None, test_size=0.2, random_state=42):
    """
    Prepare features for tabular models (LightGBM)
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns (if None, uses all except target)
        test_size: Proportion of data for testing
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test, list(X.columns)


def scale_features(X_train, X_test=None, scaler_type="standard"):
    """
    Scale features for time-series models
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        scaler_type: "standard" or "minmax"
    
    Returns:
        X_train_scaled, X_test_scaled (if provided), scaler
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Reshape for 3D time-series data
    if len(X_train.shape) == 3:
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        if X_test is not None:
            n_samples_test, n_timesteps_test, n_features_test = X_test.shape
            X_test_2d = X_test.reshape(-1, n_features_test)
            X_test_scaled = scaler.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps_test, n_features_test)
            return X_train_scaled, X_test_scaled, scaler
        return X_train_scaled, None, scaler
    else:
        # 2D data (tabular)
        X_train_scaled = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        return X_train_scaled, None, scaler


def calculate_time_series_metrics(y_true, y_pred):
    """
    Calculate metrics for time-series predictions
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics (MAE, RMSE, MAPE)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Convert to numpy arrays and flatten
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN and Inf values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if valid_mask.sum() == 0:
        print("Warning: No valid values for metrics calculation (all NaN/Inf)")
        return {
            "mae": None,
            "rmse": None,
            "mape": None
        }
    
    if valid_mask.sum() < len(y_true):
        print(f"Warning: Removed {len(y_true) - valid_mask.sum()} invalid values (NaN/Inf) from metrics calculation")
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        return {
            "mae": None,
            "rmse": None,
            "mape": None
        }
    
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    
    # MAPE (Mean Absolute Percentage Error) - handle division by zero
    mask = y_true_valid != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_valid[mask] - y_pred_valid[mask]) / y_true_valid[mask])) * 100
    else:
        mape = np.nan
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else None
    }
