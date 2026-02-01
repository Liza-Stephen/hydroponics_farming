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
     - Cluster: GPU cluster recommended
   - **Task 3: train_gru** (depends on create_feature_store)
     - Type: Python script
     - Path: `src/ml/training/train_gru.py`
     - Parameters: `{catalog}`, `{s3_bucket}`, `{catalog}.feature_store.sensor_features`, `ph_level`
     - Cluster: GPU cluster recommended
   - **Task 4: train_lightgbm** (depends on create_feature_store)
     - Type: Python script
     - Path: `src/ml/training/train_lightgbm.py`
     - Parameters: `{catalog}`, `{s3_bucket}`, `{catalog}.feature_store.sensor_features`, `is_env_optimal`
     - Cluster: CPU cluster is sufficient

### Step 3: Run ML Training Job

**First Run (Create Feature Store and Train Models):**

**Via UI:**
1. Go to **Workflows** → **Jobs**
2. Find your `ML Training Pipeline` job
3. Click **Run Now**
4. Set job parameters:
   ```json
   {
     "DATABRICKS_CATALOG": "hydroponics",
     "S3_BUCKET": "hydroponics-data"
   }
   ```
5. Click **Run**

**Via CLI:**
```bash
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "hydroponics",
    "S3_BUCKET": "hydroponics-data"
  }
}'
```

**Job Execution Flow:**
1. **create_feature_store** task runs first
   - Creates `{catalog}.feature_store` database
   - Creates `sensor_features` feature table
   - Computes all engineered features from Gold layer
2. **train_lstm**, **train_gru**, and **train_lightgbm** run in parallel (after feature store completes)
   - Each model trains independently
   - Models are registered in MLflow automatically

### Step 4: Verify Setup

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
- Each model should have version 1 in "None" stage

**4. Verify Job Run:**
- Go to **Workflows** → **Jobs** → Your job → **Runs**
- Check that all tasks completed successfully
- Review logs for any warnings or errors

### Step 5: Update Feature Store (Ongoing)

After the initial setup, update the Feature Store when new data arrives:

**Option 1: Re-run Feature Store Task**
```bash
# Run only the feature store task
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "hydroponics",
    "S3_BUCKET": "hydroponics-data"
  }
}' --task-keys create_feature_store
```

**Option 2: Schedule Feature Store Updates**
- Add a schedule to your job to run feature store task daily
- Or create a separate job that only runs feature store updates

**Option 3: Programmatic Update**
```python
from ml.feature_store import create_feature_store

# Update feature store
create_feature_store(
    catalog="hydroponics",
    gold_table_name="hydroponics.gold.iot_data",
    feature_table_name="sensor_features"
)
```

### Step 6: Configure Model Retraining (Optional)

**Schedule Regular Retraining:**
1. Go to your ML training job
2. Click **Settings** → **Schedule**
3. Configure schedule (e.g., weekly, monthly)
4. Set job parameters

**Manual Retraining:**
- Re-run the entire job to retrain all models
- Or run individual training tasks to retrain specific models

### Troubleshooting Setup

**Issue: Feature Store Creation Fails**
- **Error**: "Gold table not found"
- **Solution**: Ensure Gold layer pipeline has completed successfully

**Issue: Training Fails with "Not enough data"**
- **Error**: "Need at least {sequence_length} rows"
- **Solution**: Ensure Gold layer has sufficient data (recommended: 1000+ records)

**Issue: Model Registration Fails**
- **Error**: "MLflow experiment not found"
- **Solution**: MLflow is automatically available in Databricks, check workspace permissions

**Issue: Serverless Compute Not Available**
- **Error**: "Serverless compute not enabled"
- **Solution**: 
  - Enable serverless compute in your Databricks workspace (admin required)
  - Or switch to traditional clusters by replacing `"compute"` with `"new_cluster"` in job JSON

**Issue: Training is Slow (Serverless CPU)**
- **Observation**: LSTM/GRU training takes a long time
- **Solution**: 
  - This is expected on serverless CPU - training will complete but slower than GPU
  - Consider reducing `epochs` or `batch_size` for faster iteration
  - For production training, consider using traditional GPU clusters if available

**Issue: Repo Path Not Found**
- **Error**: "File not found: /Repos/..."
- **Solution**: Update all repo paths in `jobs/ml_training.json` to match your actual repo path

### Next Steps

After successful setup:
1. **Review Models**: Check model metrics in MLflow UI
2. **Promote Models**: Move best models to Production stage
3. **Set Up Inference**: Configure inference jobs or notebooks
4. **Monitor Performance**: Set up monitoring for production models
5. **Schedule Updates**: Configure regular feature store updates and model retraining

## Feature Store

### Overview

The Feature Store creates engineered features from the Gold layer fact table, providing a centralized repository for ML features.

### Features Created

**Raw Sensor Readings:**
- `ph_level`: pH level
- `tds_level`: Total Dissolved Solids
- `water_level`: Water level
- `air_temperature`: Air temperature (DHT sensor)
- `air_humidity`: Air humidity (DHT sensor)
- `water_temperature`: Water temperature

**Equipment State Features:**
- `ph_reducer_state`: pH reducer on/off (0/1)
- `add_water_state`: Water addition system on/off (0/1)
- `nutrients_adder_state`: Nutrient addition system on/off (0/1)
- `humidifier_state`: Humidifier on/off (0/1)
- `ex_fan_state`: Exhaust fan on/off (0/1)

**Optimal Condition Indicators:**
- `is_ph_optimal`: pH in optimal range (5.5-6.5)
- `is_tds_optimal`: TDS in optimal range (800-1200)
- `is_temp_optimal`: Temperature in optimal range (20-28°C)
- `is_humidity_optimal`: Humidity in optimal range (40-70%)

**Lag Features:**
- `ph_lag_1`: Previous pH value
- `tds_lag_1`: Previous TDS value
- `temp_lag_1`: Previous temperature value
- `humidity_lag_1`: Previous humidity value

**Rolling Statistics (1 hour):**
- `ph_avg_1h`, `ph_max_1h`, `ph_min_1h`
- `tds_avg_1h`, `tds_max_1h`, `tds_min_1h`
- `temp_avg_1h`, `temp_max_1h`, `temp_min_1h`
- `humidity_avg_1h`, `humidity_max_1h`, `humidity_min_1h`
- `water_temp_avg_1h`

**Rolling Statistics (6 hours):**
- `ph_avg_6h`, `tds_avg_6h`, `temp_avg_6h`, `humidity_avg_6h`, `water_temp_avg_6h`

### Usage

**Create Feature Store:**
```python
from ml.feature_store import create_feature_store

# Create feature store from Gold layer
feature_table = create_feature_store(
    catalog="hydroponics",
    gold_table_name="hydroponics.gold.iot_data",
    feature_table_name="sensor_features"
)
```

**Setup and Access:**

The Feature Store is automatically created when you run the ML training job or manually via the `create_feature_store()` function. Once created, the feature table is stored in the Databricks Feature Store under the specified catalog and database (default: `{catalog}.feature_store.sensor_features`).

**How it works:**
- The Feature Store reads from the Gold layer fact table (`{catalog}.gold.iot_data`)
- It creates a new database called `feature_store` in your Unity Catalog
- The feature table `sensor_features` contains all engineered features with `reading_id` as the primary key
- Features are automatically computed including lag features, rolling statistics, and optimal condition indicators
- The table is accessible via Databricks Feature Store APIs for both training and inference

**Accessing Features:**
- Training scripts automatically read from the Feature Store using `FeatureStoreManager`
- Inference scripts load features on-demand from the Feature Store
- Features are versioned and managed by Databricks Feature Store, ensuring consistency across training and inference
- The Feature Store handles feature lookups efficiently, making it ideal for real-time predictions

**Note:** The Feature Store must be created before training any models. This is typically done automatically by the ML training job, which runs the `create_feature_store` task first. See the [Setup](#setup) section for detailed setup instructions.

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

**Example:**
```python
from ml.models.lstm_model import create_lstm_model

model, optimizer, criterion = create_lstm_model(
    input_size=20,      # Number of features
    hidden_size=64,
    num_layers=2,
    output_size=1,      # Predict single value
    dropout=0.2,
    learning_rate=0.001
)
```

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

**Example:**
```python
from ml.models.gru_model import create_gru_model

model, optimizer, criterion = create_gru_model(
    input_size=20,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    dropout=0.2,
    learning_rate=0.001
)
```

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

**Example:**
```python
from ml.models.lightgbm_model import create_lightgbm_model

params, n_rounds = create_lightgbm_model(
    task_type="classification",
    num_leaves=31,
    learning_rate=0.05,
    num_boost_round=100
)
```

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

**Model Stages:**
- **None**: Newly registered model
- **Staging**: Candidate for production (testing phase)
- **Production**: Active production model
- **Archived**: Deprecated models

### Accessing Models

**Load from Registry:**
```python
from ml.utils.mlflow_utils import load_model_from_registry

# Load production model
model = load_model_from_registry("hydroponics_lstm_forecast", stage="Production")

# Load staging model
model = load_model_from_registry("hydroponics_lstm_forecast", stage="Staging")
```

**Promote Models:**
```python
from ml.utils.mlflow_utils import transition_model_stage

# Promote to production
transition_model_stage("hydroponics_lstm_forecast", version=2, stage="Production")

# Archive old model
transition_model_stage("hydroponics_lstm_forecast", version=1, stage="Archived")
```

**Get Latest Version:**
```python
from ml.utils.mlflow_utils import get_latest_model_version

version = get_latest_model_version("hydroponics_lstm_forecast")
```

### MLflow UI

Access MLflow UI in Databricks:
1. Go to **Experiments** in the left sidebar
2. Select your experiment (e.g., `hydroponics_lstm`)
3. View runs, compare metrics, and manage models

## Training Models

### Prerequisites

1. **Gold Layer Data**: Ensure Gold layer has been processed
2. **Feature Store**: Create feature store from Gold layer
3. **MLflow**: Available in Databricks (pre-installed)

### Training Workflow

#### Option 1: Using Databricks Job

**1. Create ML Training Job:**

Use the provided job configuration (`jobs/ml_training.json`):

```bash
# Create job from JSON
databricks jobs create --json-file jobs/ml_training.json
```

**2. Run Training Job:**

```bash
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "hydroponics",
    "S3_BUCKET": "hydroponics-data"
  }
}'
```

The job will:
1. Create feature store from Gold layer
2. Train LSTM model
3. Train GRU model
4. Train LightGBM model

#### Option 2: Manual Training

**1. Create Feature Store:**

```python
from ml.feature_store import create_feature_store

create_feature_store(
    catalog="hydroponics",
    gold_table_name="hydroponics.gold.iot_data",
    feature_table_name="sensor_features"
)
```

**2. Train LSTM:**

```python
from ml.training.train_lstm import train_lstm

train_lstm(
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="ph_level",
    sequence_length=24,
    forecast_horizon=1,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    experiment_name="hydroponics_lstm",
    registered_model_name="hydroponics_lstm_forecast"
)
```

**3. Train GRU:**

```python
from ml.training.train_gru import train_gru

train_gru(
    feature_table_name="hydroponics.feature_store.sensor_features",
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
)
```

**4. Train LightGBM:**

```python
from ml.training.train_lightgbm import train_lightgbm

# Classification
train_lightgbm(
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="is_ph_optimal",
    task_type="classification",
    num_leaves=31,
    learning_rate=0.05,
    num_boost_round=100,
    experiment_name="hydroponics_lightgbm",
    registered_model_name="hydroponics_lightgbm_classifier"
)

# Regression
train_lightgbm(
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="ph_level",
    task_type="regression",
    num_leaves=31,
    learning_rate=0.05,
    num_boost_round=100,
    experiment_name="hydroponics_lightgbm",
    registered_model_name="hydroponics_lightgbm_regressor"
)
```

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

### Using LSTM/GRU

**Load and Predict:**

```python
from ml.inference.predict import predict_with_lstm_gru

predictions = predict_with_lstm_gru(
    model_name="hydroponics_lstm_forecast",
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="ph_level",
    sequence_length=24,
    model_stage="Production",
    num_predictions=10
)
```

**Multi-step Forecasting:**

```python
# Predict 10 steps ahead
predictions = predict_with_lstm_gru(
    model_name="hydroponics_lstm_forecast",
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="ph_level",
    sequence_length=24,
    model_stage="Production",
    num_predictions=10  # Forecast 10 steps ahead
)
```

### Using LightGBM

**Classification:**

```python
from ml.inference.predict import predict_with_lightgbm

predictions = predict_with_lightgbm(
    model_name="hydroponics_lightgbm_classifier",
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="is_ph_optimal",
    model_stage="Production",
    num_predictions=10
)
```

**Regression:**

```python
predictions = predict_with_lightgbm(
    model_name="hydroponics_lightgbm_regressor",
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="ph_level",
    model_stage="Production",
    num_predictions=10
)
```

### Direct Model Loading

**Load PyTorch Model:**

```python
import mlflow.pytorch

model_uri = "models:/hydroponics_lstm_forecast/Production"
model = mlflow.pytorch.load_model(model_uri)
```

**Load LightGBM Model:**

```python
import mlflow.lightgbm

model_uri = "models:/hydroponics_lightgbm_classifier/Production"
model = mlflow.lightgbm.load_model(model_uri)
```

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

### Staging Workflow

**1. Train Model:**
```python
train_lstm(
    feature_table_name="...",
    registered_model_name="hydroponics_lstm_forecast"
)
# Creates version 1 in "None" stage
```

**2. Promote to Staging:**
```python
from ml.utils.mlflow_utils import transition_model_stage

transition_model_stage("hydroponics_lstm_forecast", version=1, stage="Staging")
```

**3. Test Staging Model:**
```python
# Load and test staging model
model = load_model_from_registry("hydroponics_lstm_forecast", stage="Staging")
# Run validation tests...
```

**4. Promote to Production:**
```python
# Archive current production model
transition_model_stage("hydroponics_lstm_forecast", version=1, stage="Archived")

# Promote new model to production
transition_model_stage("hydroponics_lstm_forecast", version=2, stage="Production")
```

### Reproducibility

**Model Artifacts Include:**
- Model weights/parameters
- Preprocessing scalers
- Training configuration
- Feature engineering logic
- Evaluation metrics

**Reproduce Training:**
```python
# Load model and training config from MLflow
import mlflow

run = mlflow.get_run(run_id="...")
params = run.data.params
metrics = run.data.metrics

# Recreate model with same parameters
model = create_lstm_model(
    input_size=int(params["input_size"]),
    hidden_size=int(params["hidden_size"]),
    # ... other parameters
)
```

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
│   └── predict.py                # Model inference script
└── utils/
    ├── __init__.py
    ├── mlflow_utils.py           # MLflow helper functions
    └── data_preprocessing.py     # Data preprocessing utilities
```

## Dependencies

### Pre-installed in Databricks

The following packages are pre-installed in Databricks runtime:

- **PyTorch**: For LSTM and GRU models
- **LightGBM**: For gradient boosting
- **MLflow**: For experiment tracking and model registry
- **scikit-learn**: For preprocessing and metrics
- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **databricks.feature_store**: For Feature Store

### Local Development

For local development and testing, install:

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- `torch>=2.0.0`: PyTorch for neural networks
- `lightgbm>=4.0.0`: LightGBM for tabular models
- `scikit-learn>=1.3.0`: Preprocessing and metrics
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical operations

**Note**: MLflow and Databricks Feature Store are only available in Databricks environment.

## Examples

### Example 1: Complete Training Pipeline

```python
# 1. Create feature store
from ml.feature_store import create_feature_store

feature_table = create_feature_store(
    catalog="hydroponics",
    gold_table_name="hydroponics.gold.iot_data",
    feature_table_name="sensor_features"
)

# 2. Train LSTM
from ml.training.train_lstm import train_lstm

model_lstm, scaler_X, scaler_y = train_lstm(
    feature_table_name=feature_table,
    target_col="ph_level",
    sequence_length=24,
    epochs=50
)

# 3. Train LightGBM
from ml.training.train_lightgbm import train_lightgbm

model_lgbm = train_lightgbm(
    feature_table_name=feature_table,
    target_col="is_ph_optimal",
    task_type="classification"
)
```

### Example 2: Model Inference Pipeline

```python
# 1. Load production model
from ml.utils.mlflow_utils import load_model_from_registry

model = load_model_from_registry("hydroponics_lstm_forecast", stage="Production")

# 2. Get latest features
from ml.feature_store import FeatureStoreManager

fs_manager = FeatureStoreManager("hydroponics")
df_features = fs_manager.get_feature_table("sensor_features")

# 3. Make predictions
from ml.inference.predict import predict_with_lstm_gru

predictions = predict_with_lstm_gru(
    model_name="hydroponics_lstm_forecast",
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="ph_level",
    model_stage="Production",
    num_predictions=10
)
```

### Example 3: Model Promotion Workflow

```python
from ml.utils.mlflow_utils import (
    get_latest_model_version,
    transition_model_stage,
    load_model_from_registry
)

# 1. Get latest model version
latest_version = get_latest_model_version("hydroponics_lstm_forecast")

# 2. Test staging model
staging_model = load_model_from_registry("hydroponics_lstm_forecast", stage="Staging")
# ... run validation ...

# 3. Archive old production model
transition_model_stage("hydroponics_lstm_forecast", version=latest_version-1, stage="Archived")

# 4. Promote new model to production
transition_model_stage("hydroponics_lstm_forecast", version=latest_version, stage="Production")
```

### Example 4: Custom Training Configuration

```python
from ml.training.train_lstm import train_lstm

# Custom configuration for different target
train_lstm(
    feature_table_name="hydroponics.feature_store.sensor_features",
    target_col="tds_level",  # Predict TDS instead of pH
    sequence_length=48,       # Longer sequence
    forecast_horizon=3,      # Predict 3 steps ahead
    hidden_size=128,         # Larger model
    num_layers=3,            # Deeper network
    dropout=0.3,             # More regularization
    learning_rate=0.0005,    # Lower learning rate
    batch_size=64,           # Larger batches
    epochs=100,              # More epochs
    experiment_name="hydroponics_lstm_tds",
    registered_model_name="hydroponics_lstm_tds_forecast"
)
```

## Troubleshooting

### Common Issues

**1. Feature Store Not Found:**
```
Error: Feature table not found
```
**Solution**: Create feature store first using `create_feature_store()`

**2. Model Not in Registry:**
```
Error: Model not found in registry
```
**Solution**: Train the model first, or check the model name and stage

**3. CUDA Out of Memory:**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU:
```python
device = torch.device("cpu")  # Force CPU
```

**4. Sequence Length Mismatch:**
```
Error: Input sequence length doesn't match model
```
**Solution**: Use the same `sequence_length` used during training

**5. Feature Mismatch:**
```
Error: Number of features doesn't match
```
**Solution**: Ensure feature store is up-to-date and matches training features

## Best Practices

1. **Feature Store Updates**: Update feature store regularly as new data arrives
2. **Model Monitoring**: Monitor model performance in production
3. **Version Control**: Keep track of model versions and their performance
4. **Staging Workflow**: Always test models in Staging before Production
5. **Reproducibility**: Log all hyperparameters and configurations
6. **Resource Management**: 
   - Serverless compute is recommended for simplicity (automatic resource management)
   - For faster LSTM/GRU training, use GPU clusters if available
   - Monitor training times and adjust batch sizes/epochs accordingly
7. **Data Quality**: Ensure Gold layer data quality before feature engineering

## Support

For issues or questions:
1. Check MLflow UI for experiment details
2. Review model registry for version history
3. Check Databricks job logs for training errors
4. Verify feature store is up-to-date

## Related Documentation

- [Main README](../README.md): Overall pipeline documentation
- [Databricks Feature Store](https://docs.databricks.com/machine-learning/feature-store/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
