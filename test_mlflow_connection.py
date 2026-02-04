"""
Test script to verify MLflow connection and model loading
Run this to test your Databricks MLflow setup before using the digital twin app
"""
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent / ".env"
result = load_dotenv(env_file)

# Debug: Show what was loaded
print("\nEnvironment variables after loading:")
databricks_host = os.environ.get("DATABRICKS_HOST")
databricks_token = os.environ.get("DATABRICKS_TOKEN")
catalog = os.environ.get("DATABRICKS_CATALOG")
print(f"  DATABRICKS_HOST: {databricks_host if databricks_host else 'NOT SET'}")
print(f"  DATABRICKS_TOKEN: {'SET' if databricks_token else 'NOT SET'} (length: {len(databricks_token) if databricks_token else 0})")
print(f"  DATABRICKS_CATALOG: {catalog if catalog else 'NOT SET'}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def create_lightgbm_test_input():
    """
    Create test input data for LightGBM model serving endpoint.
    
    Returns:
        list: List of feature values matching the expected LightGBM input format
    """
    # Based on feature_store.py, LightGBM expects features excluding:
    # - Raw sensor values (ph_level, tds_level, air_temperature, air_humidity)
    # - Optimal indicators (is_ph_optimal, is_tds_optimal, is_temp_optimal, is_humidity_optimal)
    # - Target column (is_env_optimal)
    # 
    # Features included:
    # - Equipment states (5): ph_reducer_state, add_water_state, nutrients_adder_state, humidifier_state, ex_fan_state
    # - Lag features (4): ph_lag_1, tds_lag_1, temp_lag_1, humidity_lag_1
    # - Rolling stats pH (4): ph_avg_1h, ph_avg_6h, ph_max_1h, ph_min_1h
    # - Rolling stats TDS (4): tds_avg_1h, tds_avg_6h, tds_max_1h, tds_min_1h
    # - Rolling stats temp (4): temp_avg_1h, temp_avg_6h, temp_max_1h, temp_min_1h
    # - Rolling stats humidity (4): humidity_avg_1h, humidity_avg_6h, humidity_max_1h, humidity_min_1h
    # - Rolling stats water temp (2): water_temp_avg_1h, water_temp_avg_6h
    # - Additional: water_level, water_temperature
    # Total: ~27-30 features (exact count depends on model training)
    
    # Create realistic test values
    test_features = [
        # Equipment states (5)
        0,  # ph_reducer_state
        0,  # add_water_state
        0,  # nutrients_adder_state
        0,  # humidifier_state
        1,  # ex_fan_state
        
        # Lag features (4)
        6.0,   # ph_lag_1
        1000,  # tds_lag_1
        22.5,  # temp_lag_1
        60.0,  # humidity_lag_1
        
        # Rolling stats - pH (4)
        6.0,   # ph_avg_1h
        6.0,   # ph_avg_6h
        6.2,   # ph_max_1h
        5.8,   # ph_min_1h
        
        # Rolling stats - TDS (4)
        1000,  # tds_avg_1h
        1000,  # tds_avg_6h
        1100,  # tds_max_1h
        900,   # tds_min_1h
        
        # Rolling stats - Temperature (4)
        22.5,  # temp_avg_1h
        22.5,  # temp_avg_6h
        23.0,  # temp_max_1h
        22.0,  # temp_min_1h
        
        # Rolling stats - Humidity (4)
        60.0,  # humidity_avg_1h
        60.0,  # humidity_avg_6h
        65.0,  # humidity_max_1h
        55.0,  # humidity_min_1h
        
        # Rolling stats - Water temperature (2)
        20.0,  # water_temp_avg_1h
        20.0,  # water_temp_avg_6h
        
        # Additional features
        70.0,  # water_level
        20.0,  # water_temperature
        
        # The model expects 30 features (indices 0-29)
        # Current count: 5 + 4 + 4 + 4 + 4 + 4 + 2 + 2 = 29
        # Need one more feature - likely a derived feature or additional rolling stat
        # Adding a placeholder - adjust based on actual model training
        0.0,   # Feature 29 - check model signature for exact feature name
    ]
    
    # Verify we have exactly 30 features
    if len(test_features) != 30:
        raise ValueError(f"Expected 30 features, but created {len(test_features)}. Model requires exactly 30 features.")
    
    return test_features


def call_model_serving(model_name, input_data):
    """
    Call Databricks model serving endpoint
    
    Args:
        model_name: Name of the model serving endpoint (can be full name like catalog.schema.model)
        input_data: list of feature values (single record)
    
    Returns:
        dict: Response from the model serving endpoint
    """
    # Extract endpoint name from model name (remove catalog.schema prefix if present)
    endpoint_name = model_name.split('.')[-1] if '.' in model_name else model_name
    
    url = f"https://{databricks_host}/serving-endpoints/{endpoint_name}/invocations"
    
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }

    # Ensure input_data is a flat list (single record) and convert to native Python types
    if isinstance(input_data[0], (int, float)):
        # Already a flat list - convert to native Python types (not numpy)
        flat_input = [float(x) for x in input_data]
    elif isinstance(input_data, list) and len(input_data) > 0:
        # If it's a list of lists, take first record
        if isinstance(input_data[0], list):
            flat_input = [float(x) for x in input_data[0]]
        else:
            flat_input = [float(x) for x in input_data]
    else:
        # Convert to list and ensure floats
        flat_input = [float(x) for x in list(input_data)]
    
    print(f"   Prepared {len(flat_input)} features as flat list")
    
    # Use dataframe_records format (standard for MLflow models)
    payload = {
        "dataframe_records": [flat_input]
    }
    
    try:
        print(f"   Calling endpoint: {url}")
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        print(f"   Success! Response received")
        return result
    except Exception as e:
        raise Exception(f"Failed to call model serving: {str(e)}")
    

def test_mlflow_connection():
    """Test MLflow connection to Databricks"""
    print("=" * 60)
    print("Testing MLflow Connection to Databricks")
    print("=" * 60)

    os.environ["DATABRICKS_HOST"] = databricks_host
    os.environ["DATABRICKS_TOKEN"] = databricks_token
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # Set tracking URI (use just "databricks" or with host)
        # For databricks-sdk, we can use just "databricks" if env vars are set
        tracking_uri = "databricks"  # This will use DATABRICKS_HOST and DATABRICKS_TOKEN from env
        mlflow.set_tracking_uri(tracking_uri)
        print(f"   MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Try to create client
        client = MlflowClient()
        
        # Try to list experiments (this tests the connection)
        print("   Testing connection by listing experiments...")
        experiments = client.search_experiments(max_results=5)
        print(f"   Connection successful! Found {len(experiments)} experiment(s)")
        
        if experiments:
            print("\n   Recent experiments:")
            for exp in experiments[:3]:
                print(f"     - {exp.name}")
        
    except Exception as e:
        print(f"   ERROR: Failed to connect to MLflow")
        print(f"   Error: {str(e)}")
        return False
    
    # Test model registry access
    print("\nTesting model registry access")
    try:
        from src.ml.utils.mlflow_utils import get_latest_model_version
        
        model_names = [
            os.environ.get('LSTM_MODEL_NAME', 'hydroponics_lstm_forecast'),
            os.environ.get('GRU_MODEL_NAME', 'hydroponics_gru_forecast'),
            os.environ.get('LIGHTGBM_MODEL_NAME', 'hydroponics_lightgbm_classifier'),
        ]
        
        print(f"   Checking for models in catalog: {catalog}")
        found_models = []
        
        for model_name in model_names:
            try:
                version, full_name = get_latest_model_version(model_name, catalog=catalog)
                if version:
                    print(f"   Found {full_name} (version {version})")
                    found_models.append((full_name, version))
                else:
                    print(f"   Model '{model_name}' not found")
            except Exception as e:
                print(f"   Error checking '{model_name}': {str(e)}")
        
        if found_models:
            print(f"\n   Found {len(found_models)} model(s) in registry")
        else:
            print("\n  No models found in registry")
        
    except Exception as e:
        print(f"   ERROR: Failed to access model registry")
        print(f"   Error: {str(e)}")
        return False
    
    # Test model loading
    print("\nTesting model loading")
    for model_name, version in found_models:
        if model_name == 'hydroponics.default.hydroponics_lightgbm_classifier':
            test_input = create_lightgbm_test_input()
            call_model_serving(model_name, test_input)
        else:
            try:
                from src.ml.utils.mlflow_utils import load_model_from_registry
                
                print(f"   Attempting to load: {model_name} (version {version})...")
                
                model = load_model_from_registry(model_name, version=version, catalog=catalog)
                print(f"   Successfully loaded model!")
                print(f"   Model type: {type(model).__name__}")
                
            except Exception as e:
                print(f"   ERROR: Failed to load model")
                print(f"   Error: {str(e)}")
                return False
    
    print("\n" + "=" * 60)
    print("Connection Test Complete!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_mlflow_connection()
