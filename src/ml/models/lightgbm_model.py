"""
LightGBM model for tabular classification/regression
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def create_lightgbm_model(
    task_type="classification",
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_data_in_leaf=20,
    num_boost_round=100,
    **kwargs
):
    """
    Create LightGBM model for tabular data
    
    Args:
        task_type: "classification" or "regression"
        num_leaves: Number of leaves in one tree
        learning_rate: Learning rate
        feature_fraction: Fraction of features to use per tree
        bagging_fraction: Fraction of data to use per tree
        bagging_freq: Frequency for bagging
        min_data_in_leaf: Minimum data in leaf
        num_boost_round: Number of boosting rounds
        **kwargs: Additional LightGBM parameters
    
    Returns:
        LightGBM model
    """
    params = {
        "objective": "binary" if task_type == "classification" else "regression",
        "metric": "binary_logloss" if task_type == "classification" else "rmse",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "min_data_in_leaf": min_data_in_leaf,
        "verbose": -1,
        **kwargs
    }
    
    return params, num_boost_round


def train_lightgbm_model(
    X_train, y_train,
    X_val=None, y_val=None,
    params=None,
    num_boost_round=100,
    early_stopping_rounds=10,
    verbose_eval=10
):
    """
    Train LightGBM model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        params: LightGBM parameters
        num_boost_round: Number of boosting rounds
        early_stopping_rounds: Early stopping rounds
        verbose_eval: Verbosity level
    
    Returns:
        Trained model
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    
    valid_sets = [train_data]
    valid_names = ["train"]
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("valid")
    
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds) if X_val is not None else None,
            lgb.log_evaluation(verbose_eval)
        ]
    )
    
    return model


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate LightGBM classification model
    
    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    # ROC AUC (if binary classification)
    if len(set(y_test)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        except:
            pass
    
    # Warning for suspiciously perfect metrics (potential data leakage or overfitting)
    perfect_threshold = 0.9999
    if all(v >= perfect_threshold for k, v in metrics.items() if k != "roc_auc"):
        print("\n" + "="*60)
        print("WARNING: Suspiciously Perfect Metrics Detected!")
        print("="*60)
        print("All classification metrics are near-perfect (>= 0.9999).")
        print("This may indicate:")
        print("  1. Data leakage: Target or related features in training data")
        print("  2. Overfitting: Model memorized training patterns")
        print("  3. Test set issues: Too small or same as training set")
        print("  4. Target directly derivable from features")
        print("\nRecommendations:")
        print("  - Check if target column or related columns are in feature set")
        print("  - Verify train/test split is correct and independent")
        print("  - Review feature importance for suspicious patterns")
        print("  - Consider cross-validation to verify performance")
        print("="*60 + "\n")
    
    return metrics


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate LightGBM regression model
    
    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2_score": float(r2_score(y_test, y_pred))
    }
    
    return metrics
