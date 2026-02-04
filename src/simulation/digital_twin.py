"""
Digital Twin Simulation Engine for Hydroponics System
Simulates sensor readings, equipment behavior, and environment conditions
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import torch
from src.simulation.llm_processor import LLMProcessor


class HydroponicsDigitalTwin:
    """
    Digital twin that simulates a hydroponics farming system
    """
    
    # Optimal ranges
    OPTIMAL_PH = (5.5, 6.5)
    OPTIMAL_TDS = (800, 1200)  # ppm
    OPTIMAL_TEMP = (20, 25)  # Celsius
    OPTIMAL_HUMIDITY = (50, 70)  # percent
    OPTIMAL_WATER_LEVEL = (60, 80)  # percent
    
    # Equipment effects
    PH_REDUCER_EFFECT = -0.1  # per minute when on
    WATER_ADDITION_EFFECT = 50  # ml per minute
    NUTRIENT_ADDITION_EFFECT = 50  # ppm per minute
    HUMIDIFIER_EFFECT = 5  # percent per minute
    FAN_EFFECT = -2  # degrees per minute
    
    def __init__(
        self,
        initial_ph: float = 6.0,
        initial_tds: float = 1000,
        initial_air_temp: float = 22.5,
        initial_air_humidity: float = 60,
        initial_water_temp: float = 20,
        initial_water_level: float = 70,
        ml_models: Optional[Dict] = None,
        model_serving_client: Optional[object] = None,
        llm_processor: Optional[LLMProcessor] = None
    ):
        """
        Initialize digital twin
        
        Args:
            initial_ph: Initial pH level
            initial_tds: Initial TDS level (ppm)
            initial_air_temp: Initial air temperature (Celsius)
            initial_air_humidity: Initial air humidity (%)
            initial_water_temp: Initial water temperature (Celsius)
            initial_water_level: Initial water level (%)
            ml_models: Dictionary with ML models {'lstm': model, 'gru': model, 'lightgbm': model}
        """
        self.state = {
            'ph_level': initial_ph,
            'tds_level': initial_tds,
            'air_temperature': initial_air_temp,
            'air_humidity': initial_air_humidity,
            'water_temperature': initial_water_temp,
            'water_level': initial_water_level,
        }
        
        self.equipment = {
            'ph_reducer_on': False,
            'add_water_on': False,
            'nutrients_adder_on': False,
            'humidifier_on': False,
            'ex_fan_on': False,
        }
        
        self.ml_models = ml_models or {}
        self.model_serving_client = model_serving_client
        self.history = []
        self.sequence_buffer = []  # For LSTM/GRU sequence prediction
        self.sequence_length = 24
        self.llm_processor = llm_processor
        
    def set_equipment(self, equipment: Dict[str, bool]):
        """Set equipment states"""
        self.equipment.update(equipment)
    
    def get_optimality(self) -> Dict[str, bool]:
        """Check if current conditions are optimal"""
        return {
            'is_ph_optimal': self.OPTIMAL_PH[0] <= self.state['ph_level'] <= self.OPTIMAL_PH[1],
            'is_tds_optimal': self.OPTIMAL_TDS[0] <= self.state['tds_level'] <= self.OPTIMAL_TDS[1],
            'is_temp_optimal': self.OPTIMAL_TEMP[0] <= self.state['air_temperature'] <= self.OPTIMAL_TEMP[1],
            'is_humidity_optimal': self.OPTIMAL_HUMIDITY[0] <= self.state['air_humidity'] <= self.OPTIMAL_HUMIDITY[1],
        }
    
    def is_environment_optimal(self) -> bool:
        """Check if entire environment is optimal"""
        optimality = self.get_optimality()
        return all(optimality.values())
    
    def _apply_equipment_effects(self, dt_minutes: float = 1.0):
        """Apply equipment effects to system state"""
        # pH reducer
        if self.equipment['ph_reducer_on']:
            self.state['ph_level'] += self.PH_REDUCER_EFFECT * dt_minutes
            self.state['ph_level'] = max(0, min(14, self.state['ph_level']))
        
        # Water addition
        if self.equipment['add_water_on']:
            self.state['water_level'] += (self.WATER_ADDITION_EFFECT / 1000) * dt_minutes  # Convert to percentage
            self.state['water_level'] = min(100, self.state['water_level'])
            # Dilution effect on TDS
            dilution_factor = 1 - (self.WATER_ADDITION_EFFECT / 1000) * dt_minutes / 100
            self.state['tds_level'] *= max(0.9, dilution_factor)
        
        # Nutrient addition
        if self.equipment['nutrients_adder_on']:
            self.state['tds_level'] += self.NUTRIENT_ADDITION_EFFECT * dt_minutes
            self.state['tds_level'] = min(3000, self.state['tds_level'])  # Cap at 3000 ppm
        
        # Humidifier
        if self.equipment['humidifier_on']:
            self.state['air_humidity'] += self.HUMIDIFIER_EFFECT * dt_minutes
            self.state['air_humidity'] = min(100, self.state['air_humidity'])
        
        # Exhaust fan (cools and reduces humidity)
        if self.equipment['ex_fan_on']:
            self.state['air_temperature'] += self.FAN_EFFECT * dt_minutes
            self.state['air_temperature'] = max(10, self.state['air_temperature'])
            self.state['air_humidity'] -= 1 * dt_minutes
            self.state['air_humidity'] = max(0, self.state['air_humidity'])
    
    def _natural_drift(self, dt_minutes: float = 1.0):
        """Simulate natural drift in system (pH tends to rise, TDS decreases, etc.)"""
        # pH tends to drift toward neutral (7.0) slowly
        ph_drift = (7.0 - self.state['ph_level']) * 0.001 * dt_minutes
        self.state['ph_level'] += ph_drift
        
        # TDS decreases slightly over time (nutrient uptake)
        self.state['tds_level'] -= 0.5 * dt_minutes
        
        # Temperature drifts toward ambient (assume 20C ambient)
        temp_drift = (20.0 - self.state['air_temperature']) * 0.01 * dt_minutes
        self.state['air_temperature'] += temp_drift
        
        # Humidity drifts toward 50%
        humidity_drift = (50.0 - self.state['air_humidity']) * 0.01 * dt_minutes
        self.state['air_humidity'] += humidity_drift
        
        # Water level decreases (evaporation)
        self.state['water_level'] -= 0.01 * dt_minutes
        self.state['water_level'] = max(0, self.state['water_level'])
    
    def _prepare_lightgbm_features(self) -> np.ndarray:
        """Prepare features for LightGBM model (30 features expected)"""
        # Equipment states (5)
        equipment_features = [
            float(self.equipment['ph_reducer_on']),
            float(self.equipment['add_water_on']),
            float(self.equipment['nutrients_adder_on']),
            float(self.equipment['humidifier_on']),
            float(self.equipment['ex_fan_on']),
        ]
        
        # Lag features (4) - use previous value from sequence buffer if available
        if len(self.sequence_buffer) >= 2:
            prev_state = self.sequence_buffer[-2]
            lag_features = [
                prev_state[0],  # ph_lag_1
                prev_state[1],  # tds_lag_1
                prev_state[2],  # temp_lag_1
                prev_state[3],  # humidity_lag_1
            ]
        else:
            # Use current values if no history
            lag_features = [
                self.state['ph_level'],
                self.state['tds_level'],
                self.state['air_temperature'],
                self.state['air_humidity'],
            ]
        
        # Rolling statistics - compute from sequence buffer
        if len(self.sequence_buffer) >= 6:  # At least 6 time steps for 1h stats
            recent_states = np.array(self.sequence_buffer[-6:])
            # pH rolling stats (4)
            ph_values = recent_states[:, 0]
            ph_avg_1h = np.mean(ph_values)
            ph_avg_6h = np.mean(ph_values) if len(self.sequence_buffer) >= 36 else ph_avg_1h
            ph_max_1h = np.max(ph_values)
            ph_min_1h = np.min(ph_values)
            
            # TDS rolling stats (4)
            tds_values = recent_states[:, 1]
            tds_avg_1h = np.mean(tds_values)
            tds_avg_6h = np.mean(tds_values) if len(self.sequence_buffer) >= 36 else tds_avg_1h
            tds_max_1h = np.max(tds_values)
            tds_min_1h = np.min(tds_values)
            
            # Temp rolling stats (4)
            temp_values = recent_states[:, 2]
            temp_avg_1h = np.mean(temp_values)
            temp_avg_6h = np.mean(temp_values) if len(self.sequence_buffer) >= 36 else temp_avg_1h
            temp_max_1h = np.max(temp_values)
            temp_min_1h = np.min(temp_values)
            
            # Humidity rolling stats (4)
            humidity_values = recent_states[:, 3]
            humidity_avg_1h = np.mean(humidity_values)
            humidity_avg_6h = np.mean(humidity_values) if len(self.sequence_buffer) >= 36 else humidity_avg_1h
            humidity_max_1h = np.max(humidity_values)
            humidity_min_1h = np.min(humidity_values)
            
            # Water temp rolling stats (2)
            water_temp_values = recent_states[:, 4]
            water_temp_avg_1h = np.mean(water_temp_values)
            water_temp_avg_6h = np.mean(water_temp_values) if len(self.sequence_buffer) >= 36 else water_temp_avg_1h
        else:
            # Use current values if insufficient history
            ph_avg_1h = ph_avg_6h = ph_max_1h = ph_min_1h = self.state['ph_level']
            tds_avg_1h = tds_avg_6h = tds_max_1h = tds_min_1h = self.state['tds_level']
            temp_avg_1h = temp_avg_6h = temp_max_1h = temp_min_1h = self.state['air_temperature']
            humidity_avg_1h = humidity_avg_6h = humidity_max_1h = humidity_min_1h = self.state['air_humidity']
            water_temp_avg_1h = water_temp_avg_6h = self.state['water_temperature']
        
        rolling_stats = [
            ph_avg_1h, ph_avg_6h, ph_max_1h, ph_min_1h,
            tds_avg_1h, tds_avg_6h, tds_max_1h, tds_min_1h,
            temp_avg_1h, temp_avg_6h, temp_max_1h, temp_min_1h,
            humidity_avg_1h, humidity_avg_6h, humidity_max_1h, humidity_min_1h,
            water_temp_avg_1h, water_temp_avg_6h,
        ]
        
        # Additional features (2)
        additional_features = [
            self.state['water_level'],
            self.state['water_temperature'],
        ]
        
        # Combine all features: 5 + 4 + 18 + 2 = 29 features
        # Model expects 30, so add a placeholder (might be a derived feature)
        all_features = equipment_features + lag_features + rolling_stats + additional_features + [0.0]
        
        return np.array(all_features, dtype=np.float32)
    
    def _prepare_lstm_gru_features(self) -> np.ndarray:
        """Prepare full feature sequence for LSTM/GRU (37 features expected)"""
        # Models were trained on all numeric columns from feature table
        # Based on feature_store.py, the features include:
        # - reading_id, timestamp (excluded from numeric)
        # - Raw sensor readings (6): ph_level, tds_level, water_level, air_temperature, air_humidity, water_temperature
        # - Equipment states (5)
        # - Optimal indicators (4)
        # - Lag features (4)
        # - Rolling stats pH (4)
        # - Rolling stats TDS (4)
        # - Rolling stats temp (4)
        # - Rolling stats humidity (4)
        # - Rolling stats water temp (2)
        # Total: 6 + 5 + 4 + 4 + 4 + 4 + 4 + 4 + 2 = 37 features
        
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        # Get sequence of sensor readings
        seq_sensor = np.array(self.sequence_buffer[-self.sequence_length:])
        
        # For each time step, compute engineered features
        full_sequences = []
        for i in range(self.sequence_length):
            # Current state at this time step
            current = seq_sensor[i]
            
            # Raw sensor readings (6): ph, tds, water_level, air_temp, air_humidity, water_temp
            raw_sensors = list(current)
            
            # Equipment states (5)
            equipment = [
                float(self.equipment['ph_reducer_on']),
                float(self.equipment['add_water_on']),
                float(self.equipment['nutrients_adder_on']),
                float(self.equipment['humidifier_on']),
                float(self.equipment['ex_fan_on']),
            ]
            
            # Optimal indicators (4) - compute based on current values
            ph_optimal = float(5.5 <= current[0] <= 6.5)
            tds_optimal = float(800 <= current[1] <= 1200)
            temp_optimal = float(20 <= current[2] <= 25)
            humidity_optimal = float(50 <= current[3] <= 70)
            optimal = [ph_optimal, tds_optimal, temp_optimal, humidity_optimal]
            
            # Lag features (4) - previous values
            if i > 0:
                lag = list(seq_sensor[i-1][:4])
            else:
                lag = list(current[:4])
            
            # Rolling stats (compute from history up to this point)
            window_start = max(0, i - 6)
            window_data = seq_sensor[window_start:i+1]
            
            ph_vals = window_data[:, 0]
            tds_vals = window_data[:, 1]
            temp_vals = window_data[:, 2]
            humidity_vals = window_data[:, 3]
            water_temp_vals = window_data[:, 4]
            
            # pH rolling stats (4)
            ph_rolling = [
                np.mean(ph_vals),
                np.mean(ph_vals) if len(window_data) >= 36 else np.mean(ph_vals),  # 6h avg
                np.max(ph_vals),
                np.min(ph_vals),
            ]
            
            # TDS rolling stats (4)
            tds_rolling = [
                np.mean(tds_vals),
                np.mean(tds_vals) if len(window_data) >= 36 else np.mean(tds_vals),
                np.max(tds_vals),
                np.min(tds_vals),
            ]
            
            # Temp rolling stats (4)
            temp_rolling = [
                np.mean(temp_vals),
                np.mean(temp_vals) if len(window_data) >= 36 else np.mean(temp_vals),
                np.max(temp_vals),
                np.min(temp_vals),
            ]
            
            # Humidity rolling stats (4)
            humidity_rolling = [
                np.mean(humidity_vals),
                np.mean(humidity_vals) if len(window_data) >= 36 else np.mean(humidity_vals),
                np.max(humidity_vals),
                np.min(humidity_vals),
            ]
            
            # Water temp rolling stats (2)
            water_temp_rolling = [
                np.mean(water_temp_vals),
                np.mean(water_temp_vals) if len(window_data) >= 36 else np.mean(water_temp_vals),
            ]
            
            # Combine all features: 6 + 5 + 4 + 4 + 4 + 4 + 4 + 4 + 2 = 37
            full_feature = (
                raw_sensors +  # 6
                equipment +    # 5
                optimal +      # 4
                lag +          # 4
                ph_rolling +   # 4
                tds_rolling +  # 4
                temp_rolling + # 4
                humidity_rolling + # 4
                water_temp_rolling  # 2
            )
            
            full_sequences.append(full_feature)
        
        return np.array(full_sequences, dtype=np.float32)
    
    def _predict_with_ml(self, features: np.ndarray) -> Dict[str, float]:
        """Use ML models to predict next state (via model serving or local models)"""
        predictions = {}
        
        # Use model serving if available, otherwise use local models
        use_serving = self.model_serving_client is not None
        
        # LSTM/GRU prediction (sequence-based) - both loaded locally
        full_seq = self._prepare_lstm_gru_features()
        if full_seq is not None:
            # LSTM prediction (local model)
            if 'lstm' in self.ml_models:
                try:
                    import torch
                    model = self.ml_models['lstm']
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = model.to(device)
                    model.eval()
                    
                    # Reshape to (1, sequence_length, features)
                    seq_tensor = torch.FloatTensor(full_seq).unsqueeze(0).to(device)
                    
                    # Scale (simple standardization per feature)
                    seq_mean = seq_tensor.mean(dim=(0, 1), keepdim=True)
                    seq_std = seq_tensor.std(dim=(0, 1), keepdim=True) + 1e-8
                    seq_scaled = (seq_tensor - seq_mean) / seq_std
                    
                    with torch.no_grad():
                        pred = model(seq_scaled).cpu().numpy().flatten()[0]
                        # Inverse scale (use mean/std of pH feature, index 0)
                        pred = pred * seq_std[0, 0, 0].item() + seq_mean[0, 0, 0].item()
                        predictions['lstm_ph'] = pred
                except Exception as e:
                    print(f"LSTM prediction error: {e}")
            
            # GRU prediction (local model)
            if 'gru' in self.ml_models:
                try:
                    import torch
                    model = self.ml_models['gru']
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = model.to(device)
                    model.eval()
                    
                    seq_tensor = torch.FloatTensor(full_seq).unsqueeze(0).to(device)
                    
                    seq_mean = seq_tensor.mean(dim=(0, 1), keepdim=True)
                    seq_std = seq_tensor.std(dim=(0, 1), keepdim=True) + 1e-8
                    seq_scaled = (seq_tensor - seq_mean) / seq_std
                    
                    with torch.no_grad():
                        pred = model(seq_scaled).cpu().numpy().flatten()[0]
                        pred = pred * seq_std[0, 0, 0].item() + seq_mean[0, 0, 0].item()
                        predictions['gru_ph'] = pred
                except Exception as e:
                    print(f"GRU prediction error: {e}")
        
        # LightGBM prediction (tabular) - prepare proper features
        lightgbm_features = self._prepare_lightgbm_features()
        
        if use_serving:
            try:
                pred = self.model_serving_client.predict_lightgbm(lightgbm_features, "hydroponics_lightgbm_classifier")
                predictions['lightgbm_env_optimal'] = pred
            except Exception as e:
                print(f"LightGBM serving error: {e}")
        
        if 'lightgbm' in self.ml_models and 'lightgbm_env_optimal' not in predictions:
            try:
                model = self.ml_models['lightgbm']
                pred = model.predict(lightgbm_features.reshape(1, -1))[0]
                predictions['lightgbm_env_optimal'] = pred
            except Exception as e:
                print(f"LightGBM prediction error: {e}")
        
        return predictions
    
    def step(self, dt_minutes: float = 1.0, use_ml: bool = True) -> Dict:
        """
        Advance simulation by one time step
        
        Args:
            dt_minutes: Time step in minutes
            use_ml: Whether to use ML models for predictions
        
        Returns:
            Dictionary with current state, equipment, optimality, and predictions
        """
        # Apply equipment effects
        self._apply_equipment_effects(dt_minutes)
        
        # Apply natural drift
        self._natural_drift(dt_minutes)
        
        # Get optimality
        optimality = self.get_optimality()
        
        # Prepare features for ML prediction
        features = np.array([
            self.state['ph_level'],
            self.state['tds_level'],
            self.state['air_temperature'],
            self.state['air_humidity'],
            self.state['water_temperature'],
            self.state['water_level'],
            float(self.equipment['ph_reducer_on']),
            float(self.equipment['add_water_on']),
            float(self.equipment['nutrients_adder_on']),
            float(self.equipment['humidifier_on']),
            float(self.equipment['ex_fan_on']),
            float(optimality['is_ph_optimal']),
            float(optimality['is_tds_optimal']),
            float(optimality['is_temp_optimal']),
            float(optimality['is_humidity_optimal']),
        ])
        
        # Update sequence buffer (only sensor readings for LSTM/GRU)
        sensor_features = np.array([
            self.state['ph_level'],
            self.state['tds_level'],
            self.state['air_temperature'],
            self.state['air_humidity'],
            self.state['water_temperature'],
            self.state['water_level'],
        ])
        self.sequence_buffer.append(sensor_features)
        if len(self.sequence_buffer) > self.sequence_length * 2:
            self.sequence_buffer = self.sequence_buffer[-self.sequence_length * 2:]
        
        # ML predictions
        ml_predictions = {}
        if use_ml and self.ml_models:
            ml_predictions = self._predict_with_ml(features)
        
        # Create state snapshot
        snapshot = {
            'timestamp': datetime.now(),
            'state': self.state.copy(),
            'equipment': self.equipment.copy(),
            'optimality': optimality,
            'is_environment_optimal': self.is_environment_optimal(),
            'ml_predictions': ml_predictions,
        }
        
        self.history.append(snapshot)
        return snapshot
    
    def simulate(
        self,
        duration_minutes: int,
        time_step_minutes: float = 1.0,
        use_ml: bool = True
    ) -> pd.DataFrame:
        """
        Run simulation for specified duration
        
        Args:
            duration_minutes: Total simulation duration in minutes
            time_step_minutes: Time step size in minutes
            use_ml: Whether to use ML models
        
        Returns:
            DataFrame with simulation history
        """
        steps = int(duration_minutes / time_step_minutes)
        
        for i in range(steps):
            self.step(time_step_minutes, use_ml=use_ml)
        
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame"""
        if not self.history:
            return pd.DataFrame()
        
        records = []
        for snapshot in self.history:
            record = {
                'timestamp': snapshot['timestamp'],
                **snapshot['state'],
                **{f'equipment_{k}': v for k, v in snapshot['equipment'].items()},
                **{k: v for k, v in snapshot['optimality'].items()},
                'is_environment_optimal': snapshot['is_environment_optimal'],
            }
            # Add ML predictions
            for k, v in snapshot.get('ml_predictions', {}).items():
                record[f'pred_{k}'] = v
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_recommendations(self, simulation_result: Optional[pd.DataFrame] = None) -> List[str]:
        """Generate recommendations using LLM based on current state and simulation history
        
        Args:
            simulation_result: Optional DataFrame with simulation history to provide context
        """
        # Initialize LLM processor if not provided
        if self.llm_processor is None:
            try:
                self.llm_processor = LLMProcessor()
            except Exception as e:
                # Fallback to simple message if LLM is not available
                return [f"LLM not available for recommendations: {str(e)}. Please ensure Ollama is running."]
        
        optimality = self.get_optimality()
        
        # Build context for LLM
        context = f"""Current Hydroponics System State:

Sensor Readings:
- pH Level: {self.state['ph_level']:.2f} (Optimal: {optimality['is_ph_optimal']}, Range: 5.5-6.5)
- TDS Level: {self.state['tds_level']:.0f} ppm (Optimal: {optimality['is_tds_optimal']}, Range: 800-1200 ppm)
- Air Temperature: {self.state['air_temperature']:.1f}°C (Optimal: {optimality['is_temp_optimal']}, Range: 20-25°C)
- Air Humidity: {self.state['air_humidity']:.1f}% (Optimal: {optimality['is_humidity_optimal']}, Range: 50-70%)
- Water Temperature: {self.state['water_temperature']:.1f}°C
- Water Level: {self.state['water_level']:.1f}% (Optimal Range: 60-80%)

Equipment Status:
- pH Reducer: {'ON' if self.equipment['ph_reducer_on'] else 'OFF'}
- Water Addition: {'ON' if self.equipment['add_water_on'] else 'OFF'}
- Nutrients Adder: {'ON' if self.equipment['nutrients_adder_on'] else 'OFF'}
- Humidifier: {'ON' if self.equipment['humidifier_on'] else 'OFF'}
- Exhaust Fan: {'ON' if self.equipment['ex_fan_on'] else 'OFF'}

Equipment Effects (per minute):
- pH Reducer: {self.PH_REDUCER_EFFECT} pH units
- Water Addition: {self.WATER_ADDITION_EFFECT} ml
- Nutrients Adder: {self.NUTRIENT_ADDITION_EFFECT} ppm
- Humidifier: {self.HUMIDIFIER_EFFECT}%
- Exhaust Fan: {self.FAN_EFFECT}°C

Overall Environment Status: {'OPTIMAL' if self.is_environment_optimal() else 'NEEDS ATTENTION'}

Please provide specific, actionable recommendations for maintaining optimal hydroponic growing conditions. 
Include:
1. Current issues (if any) with specific values and deviations
2. Recommended equipment actions
3. Estimated correction times based on equipment effects
4. Scientific context about why these adjustments matter
5. Monitoring advice

Format as a clear, professional recommendation. If all parameters are optimal, provide positive feedback and maintenance tips."""

        try:
            # Get recommendation from LLM with system context and simulation history
            recommendation = self.llm_processor.answer_question(
                "Based on the current hydroponics system state and simulation history, provide detailed recommendations for maintaining optimal growing conditions.",
                simulation_result=simulation_result,
                system_context=context,
                include_history=False
            )
            
            # The LLM response is a single string, so we'll split it into a list
            # Try to split by common separators, or return as single item
            if recommendation:
                # Split by double newlines or numbered lists
                recommendations = [r.strip() for r in recommendation.split('\n\n') if r.strip()]
                if not recommendations:
                    recommendations = [recommendation]
                return recommendations
            else:
                return ["Unable to generate recommendations. Please try again."]
                
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}. Please ensure Ollama is running and the model is installed."]
