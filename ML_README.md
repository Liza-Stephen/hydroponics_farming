# ML Consumption Layer

Machine learning capabilities for predictive analytics on hydroponics sensor data, implementing real MLOps practices with Feature Store, MLflow, and model versioning.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Feature Store](#feature-store)
- [Models](#models)
- [MLflow Integration](#mlflow-integration)
- [Training Models](#training-models)
- [Model Inference](#model-inference)
- [Model Versioning](#model-versioning)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Examples](#examples)

## Overview

The ML consumption layer provides:

- **Feature Store**: Databricks-native feature store for centralized feature management
- **MLflow**: Experiment tracking and model registry for reproducibility
- **Time-Series Models**: LSTM and GRU for forecasting sensor values
- **Tabular Model**: LightGBM for classification/regression tasks
- **Model Versioning**: Automatic versioning with staging workflow
- **Reproducibility**: Complete tracking of parameters, metrics, and artifacts

## Architecture

### Data Flow

```
Gold Layer → Feature Store → Model Training → MLflow Registry → Inference
```

### Components

1. **Feature Store** (`src/ml/feature_store.py`)
   - Creates engineered features from Gold layer
   - Manages feature tables in Databricks
   - Provides feature lookup for training and inference

2. **Models** (`src/ml/models/`)
   - **LSTM**: Long Short-Term Memory network for time-series forecasting
   - **GRU**: Gated Recurrent Unit (lighter alternative to LSTM)
   - **LightGBM**: Gradient boosting for classification/regression tasks

3. **Training** (`src/ml/training/`)
   - Training scripts with MLflow integration
   - Automatic experiment tracking
   - Model registration

4. **Inference** (`src/ml/inference/`)
   - Model loading from MLflow registry
   - Prediction APIs for all model types

5. **Utilities** (`src/ml/utils/`)
   - MLflow helpers for experiment management
   - Data preprocessing for time-series and tabular data

## Setup

This section covers the complete setup process for the ML consumption layer, including prerequisites, Databricks job configuration, and verification steps.

### Prerequisites

Before setting up the ML layer, ensure you have:

1. **Completed Data Pipeline**: The Gold layer must be populated with data
   - Run the main data processing pipeline (Bronze → Silver → Gold)
   - Verify Gold layer tables exist: `{catalog}.gold.iot_data`, `{catalog}.gold.dim_time`, `{catalog}.gold.dim_equipment`

2. **Databricks Workspace Access**:
   - Access to Databricks workspace with Unity Catalog enabled
   - Permissions to create databases, tables, and jobs
   - Access to MLflow (included in Databricks)

3. **Databricks Repo Setup**:
   - Code must be in a Databricks Repo (not workspace files)
   - Repo path should be accessible to all job tasks


### Step 1: Verify Gold Layer Data

First, ensure your Gold layer has data:

```sql
-- Check Gold layer tables
USE CATALOG hydroponics;
SELECT COUNT(*) FROM gold.iot_data;
SELECT COUNT(*) FROM gold.dim_time;
SELECT COUNT(*) FROM gold.dim_equipment;

-- Verify data quality
SELECT 
    MIN(timestamp_key) as earliest_record,
    MAX(timestamp_key) as latest_record,
    COUNT(*) as total_records
FROM gold.iot_data;
```

**Minimum Requirements:**
- At least 1000 records for meaningful training
- Data spanning multiple days/weeks for time-series models
- Complete sensor readings (pH, TDS, temperature, humidity)

### Step 2: Create ML Training Job

1. Go to **Workflows** → **Jobs** → **Create Job**
2. Configure each task manually:
   - **Task 1: create_feature_store**
     - Type: Python script
     - Path: `src/ml/feature_store.py`
     - Parameters: `{catalog}`, `{s3_bucket}`, `{catalog}.gold.iot_data`, `sensor_features`
   - **Task 2: train_lstm** (depends on create_feature_store)
     - Type: Python script
     - Path: `src/ml/training/train_lstm.py`
     - Parameters: `{catalog}`, `{s3_bucket}`, `{catalog}.feature_store.sensor_features`, `ph_level`
   - **Task 3: train_gru** (depends on create_feature_store)
     - Type: Python script
     - Path: `src/ml/training/train_gru.py`
     - Parameters: `{catalog}`, `{s3_bucket}`, `{catalog}.feature_store.sensor_features`, `ph_level`
   - **Task 4: train_lightgbm** (depends on create_feature_store)
     - Type: Python script
     - Path: `src/ml/training/train_lightgbm.py`
     - Parameters: `{catalog}`, `{s3_bucket}`, `{catalog}.feature_store.sensor_features`, `is_env_optimal`

### Step 3: Verify Setup

**1. Check Feature Store:**
```sql
USE CATALOG hydroponics;
SHOW DATABASES IN feature_store;
SELECT COUNT(*) FROM feature_store.sensor_features;
DESCRIBE TABLE feature_store.sensor_features;
```

**2. Check MLflow Experiments:**
- Go to **Experiments** in Databricks sidebar
- You should see experiments:
  - `hydroponics_lstm`
  - `hydroponics_gru`
  - `hydroponics_lightgbm`
- Each experiment should have at least one run

**3. Check Model Registry:**
- Go to **Models** in Databricks sidebar (or MLflow UI)
- You should see registered models:
  - `hydroponics_lstm_forecast`
  - `hydroponics_gru_forecast`
  - `hydroponics_lightgbm_classifier`
- Each model should have version 1 registered

**4. Verify Job Run:**
- Go to **Workflows** → **Jobs** → Your job → **Runs**
- Check that all tasks completed successfully
- Review logs for any warnings or errors

## Models

### 1. LSTM (Long Short-Term Memory)

**Purpose**: Time-series forecasting of sensor values

**Architecture:**
- Multi-layer LSTM with configurable hidden size
- Fully connected layers for output
- Dropout for regularization

**Use Cases:**
- Predict future pH levels
- Forecast TDS values
- Predict temperature trends
- Multi-step ahead forecasting

**Key Parameters:**
- `sequence_length`: Number of time steps for input (default: 24)
- `hidden_size`: LSTM hidden units (default: 64)
- `num_layers`: Number of LSTM layers (default: 2)
- `dropout`: Dropout rate (default: 0.2)
- `forecast_horizon`: Steps ahead to predict (default: 1)

### 2. GRU (Gated Recurrent Unit)

**Purpose**: Lighter alternative to LSTM for time-series forecasting

**Architecture:**
- Multi-layer GRU (simpler than LSTM)
- Fully connected layers for output
- Dropout for regularization

**Advantages:**
- Faster training and inference
- Lower memory usage
- Similar performance to LSTM for many tasks

**Use Cases:**
- Real-time predictions
- Resource-constrained environments
- When training speed is important

**Key Parameters:**
- Same as LSTM (sequence_length, hidden_size, num_layers, dropout)

### 3. LightGBM

**Purpose**: Tabular classification and regression

**Architecture:**
- Gradient boosting with tree-based learners
- Handles mixed feature types
- Fast training and inference

**Use Cases:**
- Predict optimal condition indicators
- Classify equipment states
- Regression on sensor values
- Feature importance analysis

**Key Parameters:**
- `task_type`: "classification" or "regression"
- `num_leaves`: Number of leaves (default: 31)
- `learning_rate`: Learning rate (default: 0.05)
- `num_boost_round`: Number of boosting rounds (default: 100)

## MLflow Integration

### Experiment Tracking

All training runs are automatically logged to MLflow with:

- **Parameters**: Model hyperparameters, training configuration
- **Metrics**: Training and validation metrics (loss, MAE, RMSE, accuracy, etc.)
- **Artifacts**: Model files, scalers, training logs
- **Tags**: Model type, target column, experiment name

### Model Registry

**Automatic Registration:**
- Models are automatically registered after training
- Each training run creates a new model version
- Models are tagged with metadata

### MLflow UI

Access MLflow UI in Databricks:
1. Go to **Experiments** in the left sidebar
2. Select your experiment (e.g., `hydroponics_lstm`)
3. View runs, compare metrics, and manage models

### Training Parameters

**LSTM/GRU:**
- `sequence_length`: Input sequence length (default: 24)
- `forecast_horizon`: Steps ahead to predict (default: 1)
- `hidden_size`: Hidden units (default: 64)
- `num_layers`: Number of layers (default: 2)
- `dropout`: Dropout rate (default: 0.2)
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 32)
- `epochs`: Training epochs (default: 50)

**LightGBM:**
- `task_type`: "classification" or "regression"
- `num_leaves`: Number of leaves (default: 31)
- `learning_rate`: Learning rate (default: 0.05)
- `num_boost_round`: Boosting rounds (default: 100)
- `feature_fraction`: Feature sampling (default: 0.9)
- `bagging_fraction`: Data sampling (default: 0.8)

## Model Inference

### Batch Inference

The `batch_inference.py` script loads the latest versions of all models (LSTM, GRU, LightGBM) and performs batch predictions on the entire feature table, writing results to a Delta table.

The `ml_inference.json` job configuration runs batch inference using `batch_inference.py`:

- **Parameters:**
1. `DATABRICKS_CATALOG` - Unity Catalog name (e.g., `hydroponics`)
2. `S3_BUCKET` - S3 bucket name (e.g., `hydroponics-data`)
3. `FEATURE_TABLE_NAME` - Feature table name (e.g., `feature_store.sensor_features`)
4. `LSTM_MODEL_NAME` - LSTM model name in registry
5. `GRU_MODEL_NAME` - GRU model name in registry
6. `LIGHTGBM_MODEL_NAME` - LightGBM model name in registry
7. `OUTPUT_TABLE_NAME` - Output table for predictions (e.g., `hydroponics.predictions.batch_predictions`)
8. `SEQUENCE_LENGTH` (optional) - Sequence length for LSTM/GRU (default: 24)
9. `TARGET_COL` (optional) - Target column name (default: `ph_level`)

**Output:**
The script writes predictions to the specified Delta table with the following columns:
- All original feature columns
- `lstm_prediction` - LSTM model predictions
- `gru_prediction` - GRU model predictions
- `lightgbm_prediction` - LightGBM model predictions
- `ensemble_avg` - Average of all three model predictions
- `inference_timestamp` - Timestamp when inference was run

## Model Versioning

### Version Management

**Automatic Versioning:**
- Each training run creates a new model version
- Versions are numbered sequentially (1, 2, 3, ...)
- All versions are retained in the registry

**Model Metadata:**
- Training timestamp
- Experiment name and run ID
- Hyperparameters
- Evaluation metrics
- Model artifacts (weights, scalers, configs)

## Project Structure

```
src/ml/
├── __init__.py
├── feature_store.py              # Feature Store creation and management
├── models/
│   ├── __init__.py
│   ├── lstm_model.py             # LSTM model architecture
│   ├── gru_model.py              # GRU model architecture
│   └── lightgbm_model.py         # LightGBM model
├── training/
│   ├── __init__.py
│   ├── train_lstm.py             # LSTM training script
│   ├── train_gru.py              # GRU training script
│   └── train_lightgbm.py         # LightGBM training script
├── inference/
│   ├── __init__.py
│   └── batch_inference.py         # Batch inference script (loads latest models, writes to Delta table)
└── utils/
    ├── __init__.py
    ├── mlflow_utils.py           # MLflow helper functions
    └── data_preprocessing.py     # Data preprocessing utilities
```

## Related Documentation

- [Main README](../README.md): Overall pipeline documentation
- [Databricks Feature Store](https://docs.databricks.com/machine-learning/feature-store/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
