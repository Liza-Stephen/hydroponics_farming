"""
Databricks Model Serving Client
Calls Databricks model serving endpoints instead of loading models locally
"""
import os
import requests
import numpy as np
from typing import Dict, Optional, List


class DatabricksModelServing:
    """
    Client for calling Databricks model serving endpoints
    """
    
    def __init__(
        self,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None
    ):
        """
        Initialize model serving client
        
        Args:
            databricks_host: Databricks workspace host (e.g., workspace.cloud.databricks.com)
            databricks_token: Databricks access token
        """
        self.host = databricks_host or os.environ.get("DATABRICKS_HOST")
        self.token = databricks_token or os.environ.get("DATABRICKS_TOKEN")
        
        if not self.host:
            raise ValueError("DATABRICKS_HOST not set. Set it in .env file or pass as parameter.")
        if not self.token:
            raise ValueError("DATABRICKS_TOKEN not set. Set it in .env file or pass as parameter.")
        
        # Remove protocol if present
        self.host = self.host.replace("https://", "").replace("http://", "")
        
        # Base URL for model serving
        self.base_url = f"https://{self.host}"
        
        # Headers for authentication
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def get_model_endpoint_url(self, model_name: str, version: Optional[str] = None) -> str:
        """
        Get the serving endpoint URL for a model
        
        Args:
            model_name: Model name (e.g., 'hydroponics_lstm_forecast' or 'catalog.schema.model_name')
            version: Model version (optional, uses latest if not specified)
        
        Returns:
            Endpoint URL
        """
        # Extract endpoint name from model name (remove catalog.schema prefix if present)
        endpoint_name = model_name.split('.')[-1] if '.' in model_name else model_name
        
        # Construct serving endpoint URL
        # Format: https://{host}/serving-endpoints/{endpoint_name}/invocations
        endpoint_url = f"{self.base_url}/serving-endpoints/{endpoint_name}/invocations"
        
        return endpoint_url
    
    def predict(
        self,
        model_name: str,
        inputs: List[Dict] or np.ndarray or List[List[float]],
        version: Optional[str] = None
    ) -> np.ndarray:
        """
        Make prediction using model serving endpoint
        
        Args:
            model_name: Model name (can be full name like catalog.schema.model)
            inputs: Input data (can be numpy array or list of lists)
            version: Model version (optional)
        
        Returns:
            Predictions as numpy array
        """
        endpoint_url = self.get_model_endpoint_url(model_name, version)
        
        # Convert inputs to flat list of lists format
        if isinstance(inputs, np.ndarray):
            # Convert numpy array to list of lists
            if inputs.ndim == 1:
                flat_input = [float(x) for x in inputs.tolist()]
                inputs_list = [flat_input]
            elif inputs.ndim == 2:
                inputs_list = [[float(x) for x in row] for row in inputs.tolist()]
            else:
                # Flatten or reshape as needed
                inputs_reshaped = inputs.reshape(-1, inputs.shape[-1])
                inputs_list = [[float(x) for x in row] for row in inputs_reshaped.tolist()]
        elif isinstance(inputs, list):
            # Ensure it's a list of lists and convert to native Python types
            if len(inputs) > 0 and isinstance(inputs[0], (int, float)):
                # Single flat list - wrap it
                inputs_list = [[float(x) for x in inputs]]
            elif len(inputs) > 0 and isinstance(inputs[0], list):
                # List of lists
                inputs_list = [[float(x) for x in row] for row in inputs]
            else:
                inputs_list = [[float(x) for x in inputs]]
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        # Use dataframe_records format (standard for MLflow models)
        payload = {
            "dataframe_records": inputs_list
        }
        
        try:
            response = requests.post(
                endpoint_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Parse response - format varies by model type
            # Common formats:
            # - {"predictions": [...]}
            # - {"predictions": [[...], [...]]}
            # - Direct array
            
            if "predictions" in result:
                predictions = result["predictions"]
            elif isinstance(result, list):
                predictions = result
            else:
                # Try to extract predictions from other possible keys
                predictions = result.get("outputs", result.get("values", result))
            
            # Convert to numpy array
            if isinstance(predictions, list):
                predictions = np.array(predictions)
            
            # Flatten if needed
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            return predictions
            
        except requests.exceptions.HTTPError as e:
            error_text = ""
            try:
                error_json = e.response.json() if hasattr(e.response, 'json') else {}
                error_text = error_json.get('message', e.response.text[:500]) if error_json else e.response.text[:500]
            except:
                error_text = str(e.response.text[:500]) if hasattr(e.response, 'text') else str(e)
            
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model serving endpoint not found for '{model_name}'. "
                    f"Make sure the endpoint is deployed in Databricks. "
                    f"URL: {endpoint_url}"
                )
            else:
                raise ValueError(f"Error calling model serving endpoint (HTTP {e.response.status_code}): {error_text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to connect to model serving endpoint: {str(e)}")
    
    def predict_lightgbm(
        self,
        features: np.ndarray,
        model_name: str = "hydroponics_lightgbm_classifier"
    ) -> float:
        """
        Predict using LightGBM model via serving endpoint
        
        Args:
            features: Feature array (shape: [n_features] or [1, n_features])
            model_name: Model name
        
        Returns:
            Prediction value
        """
        # Ensure 1D or 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Convert to list of lists
        inputs = features.tolist()
        
        predictions = self.predict(model_name, inputs)
        
        # Return single value
        if isinstance(predictions, np.ndarray):
            return float(predictions[0] if predictions.ndim > 0 else predictions)
        return float(predictions)
