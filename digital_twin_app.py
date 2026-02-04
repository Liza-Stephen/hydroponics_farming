"""
Streamlit App for Hydroponics Farm Simulation and BI Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mlflow
    from src.simulation.digital_twin import HydroponicsDigitalTwin
    from src.simulation.llm_processor import LLMProcessor
    from src.simulation.databricks_model_serving import DatabricksModelServing
    from src.ml.utils.mlflow_utils import load_model_from_registry, get_latest_model_version
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all dependencies are installed.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Hydroponics Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'ml_models_loaded' not in st.session_state:
    st.session_state.ml_models_loaded = False
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'llm_processor' not in st.session_state:
    st.session_state.llm_processor = None


def configure_mlflow():
    """Configure MLflow connection (match test_mlflow_connection.py pattern)"""
    databricks_host = os.environ.get("DATABRICKS_HOST")
    databricks_token = os.environ.get("DATABRICKS_TOKEN")
    
    if not databricks_host or not databricks_token:
        return False
    
    # Set environment variables before setting tracking URI
    os.environ["DATABRICKS_HOST"] = databricks_host
    os.environ["DATABRICKS_TOKEN"] = databricks_token
    mlflow.set_tracking_uri("databricks")
    return True


def load_lstm_gru_models(silent=True):
    """Load LSTM and GRU models from MLflow registry"""
    if not configure_mlflow():
        return {}
    
    models = {}
    catalog = os.environ.get('DATABRICKS_CATALOG', 'hydroponics')
    
    # Load LSTM
    if 'lstm' not in st.session_state.ml_models:
        try:
            lstm_model_name = os.environ.get('LSTM_MODEL_NAME', 'hydroponics_lstm_forecast')
            lstm_model = load_model_from_registry(lstm_model_name, catalog=catalog)
            st.session_state.ml_models['lstm'] = lstm_model
        except Exception:
            pass
    
    # Load GRU
    if 'gru' not in st.session_state.ml_models:
        try:
            gru_model_name = os.environ.get('GRU_MODEL_NAME', 'hydroponics_gru_forecast')
            gru_model = load_model_from_registry(gru_model_name, catalog=catalog)
            st.session_state.ml_models['gru'] = gru_model
        except Exception:
            pass
    
    if st.session_state.ml_models:
        st.session_state.ml_models_loaded = True
        models = {k: v for k, v in st.session_state.ml_models.items() if k in ['lstm', 'gru']}
    
    return models


def create_time_series_chart(df: pd.DataFrame, title: str):
    """Create time series chart"""
    fig = go.Figure()
    
    # Sensor readings
    if 'ph_level' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ph_level'],
            name='pH Level',
            line=dict(color='blue', width=2)
        ))
    
    if 'tds_level' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['tds_level'],
            name='TDS (ppm)',
            yaxis='y2',
            line=dict(color='green', width=2)
        ))
    
    if 'air_temperature' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['air_temperature'],
            name='Temperature (°C)',
            yaxis='y3',
            line=dict(color='red', width=2)
        ))
    
    if 'air_humidity' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['air_humidity'],
            name='Humidity (%)',
            yaxis='y4',
            line=dict(color='orange', width=2)
        ))
    
    # Add optimal ranges
    if 'ph_level' in df.columns:
        fig.add_hrect(
            y0=5.5, y1=6.5,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Optimal pH Range",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis=dict(title="pH Level", side="left"),
        yaxis2=dict(title="TDS (ppm)", overlaying="y", side="right"),
        yaxis3=dict(title="Temperature (°C)", overlaying="y", side="right", position=0.95),
        yaxis4=dict(title="Humidity (%)", overlaying="y", side="right", position=0.9),
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# Main App
st.title("Hydroponics Copilot")

# Create tabs
tab1, tab2 = st.tabs(["Simulation", "BI Dashboard"])

# Sidebar (shared across tabs)
with st.sidebar:
    st.header("Configuration")
    
    # ML Models
    # Single checkbox to load all models
    use_ml = st.checkbox("Use ML Predictions", value=False,
                         help="LSTM and GRU models loaded locally, LightGBM model served via Databricks Serving")
    
    # Auto-load models when checkbox is enabled
    if use_ml:
        # Load LSTM and GRU locally
        if 'lstm' not in st.session_state.ml_models or 'gru' not in st.session_state.ml_models:
            load_lstm_gru_models()
    
    # Scenario Selection
    st.subheader("Scenarios")
    scenario = st.selectbox(
        "Choose a scenario",
        ["Custom", "Optimal Conditions", "Stress Test", "High pH", "Low pH", 
         "Low TDS", "High TDS", "Hot", "Cold", "Dry", "Humid"]
    )
    
    # Determine if controls should be disabled
    is_custom = (scenario == "Custom")
    
    # Initial Conditions
    st.subheader("Initial Conditions")
    col1, col2 = st.columns(2)
    
    with col1:
        initial_ph = st.slider("pH Level", 0.0, 14.0, 6.0, 0.1, disabled=not is_custom)
        initial_tds = st.slider("TDS (ppm)", 0, 3000, 1000, 50, disabled=not is_custom)
        initial_temp = st.slider("Air Temp (°C)", 10.0, 35.0, 22.5, 0.5, disabled=not is_custom)
    
    with col2:
        initial_humidity = st.slider("Humidity (%)", 0, 100, 60, 5, disabled=not is_custom)
        initial_water_temp = st.slider("Water Temp (°C)", 10.0, 30.0, 20.0, 0.5, disabled=not is_custom)
        initial_water_level = st.slider("Water Level (%)", 0, 100, 70, 5, disabled=not is_custom)
    
    # Equipment States
    st.subheader("Equipment")
    ph_reducer = st.checkbox("pH Reducer", value=False, disabled=not is_custom)
    add_water = st.checkbox("Add Water", value=False, disabled=not is_custom)
    nutrients = st.checkbox("Nutrients Adder", value=False, disabled=not is_custom)
    humidifier = st.checkbox("Humidifier", value=False, disabled=not is_custom)
    fan = st.checkbox("Exhaust Fan", value=False, disabled=not is_custom)
    
    # Show scenario info if not custom
    if not is_custom:
        st.info(f"**{scenario}** scenario will set initial conditions automatically.")
    
    # Simulation Parameters
    st.subheader("Simulation Parameters")
    duration = st.slider("Duration (minutes)", 10, 480, 60, 10)
    time_step = st.slider("Time Step (minutes)", 0.5, 5.0, 1.0, 0.5)
    
    # Scenario conditions mapping
    SCENARIO_CONDITIONS = {
        'optimal_conditions': {
            'ph_level': 6.0,
            'tds_level': 1000,
            'air_temperature': 22.5,
            'air_humidity': 60,
            'water_temperature': 20,
            'water_level': 70,
        },
        'stress_test': {
            'ph_level': 4.0,
            'tds_level': 500,
            'air_temperature': 30,
            'air_humidity': 30,
            'water_temperature': 25,
            'water_level': 40,
        },
        'high_ph': {
            'ph_level': 8.0,
            'tds_level': 1000,
            'air_temperature': 22.5,
            'air_humidity': 60,
            'water_temperature': 20,
            'water_level': 70,
        },
        'low_ph': {
            'ph_level': 4.5,
            'tds_level': 1000,
            'air_temperature': 22.5,
            'air_humidity': 60,
            'water_temperature': 20,
            'water_level': 70,
        },
        'low_tds': {
            'ph_level': 6.0,
            'tds_level': 400,
            'air_temperature': 22.5,
            'air_humidity': 60,
            'water_temperature': 20,
            'water_level': 70,
        },
        'high_tds': {
            'ph_level': 6.0,
            'tds_level': 2000,
            'air_temperature': 22.5,
            'air_humidity': 60,
            'water_temperature': 20,
            'water_level': 70,
        },
        'hot': {
            'ph_level': 6.0,
            'tds_level': 1000,
            'air_temperature': 30,
            'air_humidity': 60,
            'water_temperature': 25,
            'water_level': 70,
        },
        'cold': {
            'ph_level': 6.0,
            'tds_level': 1000,
            'air_temperature': 15,
            'air_humidity': 60,
            'water_temperature': 15,
            'water_level': 70,
        },
        'dry': {
            'ph_level': 6.0,
            'tds_level': 1000,
            'air_temperature': 22.5,
            'air_humidity': 30,
            'water_temperature': 20,
            'water_level': 70,
        },
        'humid': {
            'ph_level': 6.0,
            'tds_level': 1000,
            'air_temperature': 22.5,
            'air_humidity': 85,
            'water_temperature': 20,
            'water_level': 70,
        },
    }
    
    # Run Simulation
    if st.button("Run Simulation", type="primary"):
        # Get scenario conditions if not custom
        if scenario != "Custom":
            scenario_key = scenario.lower().replace(" ", "_")
            scenario_conditions = SCENARIO_CONDITIONS.get(scenario_key, SCENARIO_CONDITIONS['optimal_conditions'])
            initial_ph = scenario_conditions.get('ph_level', initial_ph)
            initial_tds = scenario_conditions.get('tds_level', initial_tds)
            initial_temp = scenario_conditions.get('air_temperature', initial_temp)
            initial_humidity = scenario_conditions.get('air_humidity', initial_humidity)
            initial_water_temp = scenario_conditions.get('water_temperature', initial_water_temp)
            initial_water_level = scenario_conditions.get('water_level', initial_water_level)
        
        # Initialize twin with hybrid approach: serving for LightGBM, local for LSTM/GRU
        models = {}
        model_serving_client = None
        
        if use_ml:
            # Load LSTM and GRU locally if not already loaded
            if 'lstm' not in st.session_state.ml_models or 'gru' not in st.session_state.ml_models:
                load_lstm_gru_models()
            
            models = {k: v for k, v in st.session_state.ml_models.items() if k in ['lstm', 'gru']}
            
            # Use model serving for LightGBM
            try:
                model_serving_client = DatabricksModelServing()
            except Exception:
                model_serving_client = None
        
        # Get LLM processor if available
        llm_processor = st.session_state.llm_processor if 'llm_processor' in st.session_state else None
        
        twin = HydroponicsDigitalTwin(
            initial_ph=initial_ph,
            initial_tds=initial_tds,
            initial_air_temp=initial_temp,
            initial_air_humidity=initial_humidity,
            initial_water_temp=initial_water_temp,
            initial_water_level=initial_water_level,
            ml_models=models,
            model_serving_client=model_serving_client,
            llm_processor=llm_processor
        )
        
        # Set equipment
        twin.set_equipment({
            'ph_reducer_on': ph_reducer,
            'add_water_on': add_water,
            'nutrients_adder_on': nutrients,
            'humidifier_on': humidifier,
            'ex_fan_on': fan,
        })
        
        # Run simulation
        with st.spinner(f"Running simulation for {duration} minutes..."):
            df = twin.simulate(
                duration_minutes=duration,
                time_step_minutes=time_step,
                use_ml=use_ml and len(models) > 0
            )
        
        st.session_state.simulation_history = df
        
        # Generate recommendations after simulation completes
        if len(df) > 0:
            final_state = df.iloc[-1]
            # Update twin state to match final simulation state
            twin.state['ph_level'] = final_state['ph_level']
            twin.state['tds_level'] = final_state['tds_level']
            twin.state['air_temperature'] = final_state['air_temperature']
            twin.state['air_humidity'] = final_state['air_humidity']
            twin.state['water_temperature'] = final_state['water_temperature']
            twin.state['water_level'] = final_state['water_level']
            
            # Set equipment state from final state if available
            equipment_cols = {
                'equipment_ph_reducer_on': 'ph_reducer_on',
                'equipment_add_water_on': 'add_water_on',
                'equipment_nutrients_adder_on': 'nutrients_adder_on',
                'equipment_humidifier_on': 'humidifier_on',
                'equipment_ex_fan_on': 'ex_fan_on'
            }
            for col_name, eq_name in equipment_cols.items():
                if col_name in final_state:
                    twin.equipment[eq_name] = bool(final_state.get(col_name, False))
            
            # Generate and store recommendations
            with st.spinner("Generating recommendations..."):
                recommendations = twin.get_recommendations(simulation_result=df)
                st.session_state.recommendations = recommendations

# Simulation Tab Content
with tab1:
    # Q&A Section
    st.markdown("Explore what-if simulations, ask questions and recommendations.")
    
    # Clear conversation history button
    if st.button("Clear Conversation History"):
        if st.session_state.llm_processor:
            st.session_state.llm_processor.clear_history()
        st.success("Conversation history cleared!")
    
    # Text input
    qa_query = st.text_input(
        "Ask a question",
        placeholder="e.g., 'What is pH?', 'What is the current pH?', 'What should I do?', 'Explain TDS'",
        key="qa_input"
    )
    
    if st.button("Ask", key="qa_button"):
        if not qa_query or qa_query.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Initialize LLM processor if not already initialized
            if st.session_state.llm_processor is None:
                try:
                    current_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
                    st.session_state.llm_processor = LLMProcessor(model=current_model)
                except Exception as e:
                    st.error(f"Failed to initialize Ollama: {str(e)}")
                    st.info("""
                    **Setup Instructions:**
                    1. Install Ollama from https://ollama.ai
                    2. Start Ollama: `ollama serve` (or it may start automatically)
                    3. Pull the model: `ollama pull llama3.2`
                    4. Verify it's running: `ollama list`
                    """)
                    st.stop()
            
            simulation_data = st.session_state.simulation_history if st.session_state.simulation_history is not None else None
            
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.llm_processor.answer_question(
                        qa_query, 
                        simulation_data,
                        include_history=True
                    )
                    
                    if answer:
                        st.markdown("### Answer")
                        st.markdown(answer)
                    else:
                        st.warning("I couldn't generate a response. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                    st.info("Please check your API key and try again.")
    
    st.divider()

    # Display Results
    if st.session_state.simulation_history is not None:
        df = st.session_state.simulation_history
        
        st.header("Simulation Results")
        
        # Current State
        if len(df) > 0:
            final_state = df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("pH Level", f"{final_state['ph_level']:.2f}")
                is_optimal = final_state.get('is_ph_optimal', False)
                st.caption("Optimal" if is_optimal else "Out of Range")
            
            with col2:
                st.metric("TDS (ppm)", f"{final_state['tds_level']:.0f}")
                is_optimal = final_state.get('is_tds_optimal', False)
                st.caption("Optimal" if is_optimal else "Out of Range")
            
            with col3:
                st.metric("Temperature (°C)", f"{final_state['air_temperature']:.1f}")
                is_optimal = final_state.get('is_temp_optimal', False)
                st.caption("Optimal" if is_optimal else "Out of Range")
            
            with col4:
                st.metric("Humidity (%)", f"{final_state['air_humidity']:.1f}")
                is_optimal = final_state.get('is_humidity_optimal', False)
                st.caption("Optimal" if is_optimal else "Out of Range")
            
            # Recommendations
            st.subheader("Recommendations")
            # Display stored recommendations (generated after simulation)
            if 'recommendations' in st.session_state and st.session_state.recommendations:
                combined_recommendations = "\n\n".join(st.session_state.recommendations)
                st.info(combined_recommendations)
            else:
                st.info("Run a simulation to generate recommendations.")
        
        # Charts
        st.subheader("Time Series Charts")
        
        # Main sensor chart
        fig_main = create_time_series_chart(df, "Sensor Readings Over Time")
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("Configure settings in the sidebar and run a simulation to see results.")

# BI Dashboard Tab Content
with tab2:
    try:
        from src.data_processing.snowflake.bi_queries import (
            get_daily_metrics, get_hourly_metrics, get_sensor_readings,
            get_equipment_usage_stats, get_optimality_trends
        )
        
        # Check Snowflake connection
        snowflake_configured = all([
            os.environ.get("SNOWFLAKE_ACCOUNT"),
            os.environ.get("SNOWFLAKE_USER"),
            os.environ.get("SNOWFLAKE_PASSWORD"),
            os.environ.get("SNOWFLAKE_WAREHOUSE"),
            os.environ.get("SNOWFLAKE_DATABASE"),
            os.environ.get("SNOWFLAKE_SCHEMA")
        ])
        
        if not snowflake_configured:
            st.warning(
                "Snowflake credentials not configured. Please set the following environment variables in your .env file:\n"
                "- SNOWFLAKE_ACCOUNT\n"
                "- SNOWFLAKE_USER\n"
                "- SNOWFLAKE_PASSWORD\n"
                "- SNOWFLAKE_WAREHOUSE\n"
                "- SNOWFLAKE_DATABASE\n"
                "- SNOWFLAKE_SCHEMA"
            )
        else:
            # Initialize variables
            daily_metrics = None
            daily_metrics_for_kpis = None
            hourly_metrics = None
            equipment_stats = None
            optimality_trends = None
            
            # First, get date range from data
            with st.spinner("Loading date range..."):
                daily_metrics_all = get_daily_metrics()
            
            if daily_metrics_all is not None and not daily_metrics_all.empty and 'reading_date' in daily_metrics_all.columns:
                # Get min and max dates
                min_date = pd.to_datetime(daily_metrics_all['reading_date']).min().date()
                max_date = pd.to_datetime(daily_metrics_all['reading_date']).max().date()
                
                # Set reference date for KPIs (default to 2023/12/11 - December 11, 2023)
                reference_date_default = datetime(2023, 12, 11).date()
                if reference_date_default < min_date or reference_date_default > max_date:
                    # If default date is out of range, use max_date
                    reference_date_default = max_date
                
                # Dashboard controls
                kpi_reference_date = reference_date_default
                start_date = min_date
                end_date = max_date
                
                # Validate date range
                if start_date > end_date:
                    st.error("Start date must be before or equal to end date.")
                    st.stop()
                
                # Convert dates to string format for queries
                kpi_date_str = kpi_reference_date.strftime('%Y-%m-%d')
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # Load data
                # For KPIs: get data for the reference date only
                # For trends: get data for the full range (min to max, or filtered range)
                with st.spinner("Loading data from semantic layer..."):
                    # KPI data - single date
                    daily_metrics_kpi = get_daily_metrics(start_date=kpi_date_str, end_date=kpi_date_str)
                    # Trend data - full range
                    daily_metrics = get_daily_metrics(start_date=start_date_str, end_date=end_date_str)
                    hourly_metrics = get_hourly_metrics(start_date=start_date_str, end_date=end_date_str)
                    equipment_stats = get_equipment_usage_stats(start_date=start_date_str, end_date=end_date_str)
                    optimality_trends = get_optimality_trends(start_date=start_date_str, end_date=end_date_str)
                    
                    # Use KPI data for metrics, trend data for charts
                    daily_metrics_for_kpis = daily_metrics_kpi if daily_metrics_kpi is not None and not daily_metrics_kpi.empty else daily_metrics
            else:
                st.error("Unable to load date range. Please check your connection and ensure dbt models have been run.")
                daily_metrics = None
                daily_metrics_for_kpis = None
                hourly_metrics = None
                equipment_stats = None
                optimality_trends = None
            
            # Check if we have KPI data
            if daily_metrics_for_kpis is None:
                st.error("Unable to load data from Snowflake semantic layer. Please check your connection and ensure dbt models have been run.")
                st.info("Check the console/terminal for detailed error messages.")
            elif daily_metrics_for_kpis.empty:
                st.warning(f"No data found for KPI reference date {kpi_reference_date}. Try selecting a different date or ensure data exists in Snowflake.")
                # Still show trends even if KPI data is missing
                daily_metrics_for_kpis = None
            else:
                # Key Metrics (using reference date)
                st.subheader("Key Performance Indicators")
                if len(daily_metrics_for_kpis) > 0:
                    latest = daily_metrics_for_kpis.iloc[0]
                    
                    kpi1, kpi2, kpi3 = st.columns(3)
                    with kpi1:
                        health_score = latest.get('daily_health_score', 0) if 'daily_health_score' in latest else 0
                        st.metric("Daily Health Score", f"{health_score:.1f}%",
                                 delta=f"{health_score - 75:.1f}% vs target" if health_score > 0 else None,
                                 help="Average of pH, TDS, Temperature, and Humidity optimality percentages")
                    with kpi2:
                        env_efficiency = latest.get('daily_environment_efficiency_pct', 0) if 'daily_environment_efficiency_pct' in latest else 0
                        st.metric("Environment Efficiency", f"{env_efficiency:.1f}%",
                                 delta=f"{env_efficiency - 80:.1f}% vs target" if env_efficiency > 0 else None,
                                 help="Percentage of readings where all conditions (pH, TDS, temp, humidity) were optimal simultaneously")
                    with kpi3:
                        total_readings = int(latest.get('total_readings', 0))
                        st.metric("Total Readings", f"{total_readings:,}",
                                 help="Total number of sensor readings for the most recent day")
                    
                    # Diagnostic information
                    with st.expander("Metric Explanations & Diagnostics"):
                        st.write("**Daily Health Score (50.0%)**:")
                        st.write("- Formula: Average of (pH optimality + TDS optimality + Temperature optimality + Humidity optimality) / 4")
                        st.write("- Current: 50% means on average, only 50% of readings had optimal conditions for each parameter")
                        st.write("- Target: 75% or higher")
                        st.write("")
                        
                        st.write("**Environment Efficiency (0.0%)**:")
                        st.write("- Formula: (Total readings with ALL conditions optimal) / (Total readings) × 100")
                        st.write("- Current: 0% means NO readings had all 4 conditions (pH, TDS, temp, humidity) optimal simultaneously")
                        st.write("- Target: 80% or higher")
                        st.write("")
                        
                        st.write("**Total Readings (1)**:")
                        st.write("- This is very low! It suggests sparse data.")
                        st.write("- Possible causes:")
                        st.write("  - Limited data in Snowflake (only 1 reading in the most recent day)")
                        st.write("  - Data aggregation issue in dbt models")
                        st.write("  - Data hasn't been loaded recently")
                        st.write("")
                        
                        
                        # Show breakdown of latest day
                        if 'avg_ph_optimality_pct' in latest:
                            st.write("**Latest Day Breakdown:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("pH Optimality", f"{latest.get('avg_ph_optimality_pct', 0):.1f}%")
                            with col2:
                                st.metric("TDS Optimality", f"{latest.get('avg_tds_optimality_pct', 0):.1f}%")
                            with col3:
                                st.metric("Temp Optimality", f"{latest.get('avg_temp_optimality_pct', 0):.1f}%")
                            with col4:
                                st.metric("Humidity Optimality", f"{latest.get('avg_humidity_optimality_pct', 0):.1f}%")
                            
                            st.write("")
                            st.write(f"**Total Environment Optimal Readings**: {int(latest.get('total_environment_optimal', 0))} out of {total_readings}")
                            if total_readings > 0:
                                st.write(f"**Environment Optimality**: {(latest.get('total_environment_optimal', 0) / total_readings * 100):.1f}%")
                
                # Daily Trends
                st.subheader("Daily Trends")
                
                if not daily_metrics.empty and 'reading_date' in daily_metrics.columns:
                    # Sensor readings over time
                    fig_sensors = go.Figure()
                    
                    # Add traces only if columns exist
                    if 'daily_avg_ph' in daily_metrics.columns:
                        fig_sensors.add_trace(go.Scatter(
                            x=daily_metrics['reading_date'],
                            y=daily_metrics['daily_avg_ph'],
                            name='pH Level',
                            line=dict(color='blue', width=2)
                        ))
                    if 'daily_avg_tds' in daily_metrics.columns:
                        fig_sensors.add_trace(go.Scatter(
                            x=daily_metrics['reading_date'],
                            y=daily_metrics['daily_avg_tds'] / 100,
                            name='TDS (ppm/100)',
                            yaxis='y2',
                            line=dict(color='green', width=2)
                        ))
                    if 'daily_avg_air_temp' in daily_metrics.columns:
                        fig_sensors.add_trace(go.Scatter(
                            x=daily_metrics['reading_date'],
                            y=daily_metrics['daily_avg_air_temp'],
                            name='Temperature (°C)',
                            yaxis='y3',
                            line=dict(color='red', width=2)
                        ))
                    if 'daily_avg_air_humidity' in daily_metrics.columns:
                        fig_sensors.add_trace(go.Scatter(
                            x=daily_metrics['reading_date'],
                            y=daily_metrics['daily_avg_air_humidity'],
                            name='Humidity (%)',
                            yaxis='y4',
                            line=dict(color='orange', width=2)
                        ))
                    
                    if len(fig_sensors.data) > 0:
                        fig_sensors.update_layout(
                            title="Daily Average Sensor Readings",
                            xaxis_title="Date",
                            yaxis=dict(title="pH Level", side="left"),
                            yaxis2=dict(title="TDS (ppm/100)", overlaying="y", side="right"),
                            yaxis3=dict(title="Temperature (°C)", overlaying="y", side="right", position=0.95),
                            yaxis4=dict(title="Humidity (%)", overlaying="y", side="right", position=0.9),
                            hovermode='x unified',
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_sensors, use_container_width=True)
                    else:
                        st.info("Sensor data columns not available in the semantic layer.")
                elif not daily_metrics.empty:
                    st.warning(f"Date column not found. Available columns: {', '.join(daily_metrics.columns.tolist())}")
                
                # Optimality Trends
                st.subheader("Optimality Trends")
                
                if optimality_trends is not None and not optimality_trends.empty and 'reading_date' in optimality_trends.columns:
                    fig_optimality = go.Figure()
                    
                    # Add traces only if columns exist
                    if 'ph_optimality' in optimality_trends.columns:
                        fig_optimality.add_trace(go.Scatter(
                            x=optimality_trends['reading_date'],
                            y=optimality_trends['ph_optimality'],
                            name='pH Optimality',
                            line=dict(color='blue', width=2)
                        ))
                    if 'tds_optimality' in optimality_trends.columns:
                        fig_optimality.add_trace(go.Scatter(
                            x=optimality_trends['reading_date'],
                            y=optimality_trends['tds_optimality'],
                            name='TDS Optimality',
                            line=dict(color='green', width=2)
                        ))
                    if 'temp_optimality' in optimality_trends.columns:
                        fig_optimality.add_trace(go.Scatter(
                            x=optimality_trends['reading_date'],
                            y=optimality_trends['temp_optimality'],
                            name='Temperature Optimality',
                            line=dict(color='red', width=2)
                        ))
                    if 'humidity_optimality' in optimality_trends.columns:
                        fig_optimality.add_trace(go.Scatter(
                            x=optimality_trends['reading_date'],
                            y=optimality_trends['humidity_optimality'],
                            name='Humidity Optimality',
                            line=dict(color='orange', width=2)
                        ))
                    if 'environment_optimality' in optimality_trends.columns:
                        fig_optimality.add_trace(go.Scatter(
                            x=optimality_trends['reading_date'],
                            y=optimality_trends['environment_optimality'],
                            name='Overall Environment',
                            line=dict(color='purple', width=3)
                        ))
                    
                    if len(fig_optimality.data) > 0:
                        fig_optimality.update_layout(
                            title="Optimality Percentages Over Time",
                            xaxis_title="Date",
                            yaxis_title="Optimality (%)",
                            yaxis=dict(range=[0, 100]),
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig_optimality, use_container_width=True)
                    else:
                        st.info("Optimality data columns not available in the semantic layer.")
                elif optimality_trends is not None and not optimality_trends.empty:
                    st.warning(f"Date column not found in optimality trends. Available columns: {', '.join(optimality_trends.columns.tolist())}")
                
                # Equipment Usage
                st.subheader("Equipment Usage Statistics")
                
                if equipment_stats is not None and not equipment_stats.empty and 'reading_date' in equipment_stats.columns:
                    fig_equipment = go.Figure()
                    equipment_cols = ['ph_reducer', 'water_addition', 'nutrients', 'humidifier', 'exhaust_fan']
                    colors = ['blue', 'cyan', 'green', 'orange', 'red']
                    
                    for col, color in zip(equipment_cols, colors):
                        if col in equipment_stats.columns:
                            fig_equipment.add_trace(go.Bar(
                                x=equipment_stats['reading_date'],
                                y=equipment_stats[col],
                                name=col.replace('_', ' ').title(),
                                marker_color=color
                            ))
                    
                    if len(fig_equipment.data) > 0:
                        fig_equipment.update_layout(
                            title="Daily Equipment Activations",
                            xaxis_title="Date",
                            yaxis_title="Number of Activations",
                            barmode='group',
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_equipment, use_container_width=True)
                    else:
                        st.info("Equipment usage columns not available in the semantic layer.")
                elif equipment_stats is not None and not equipment_stats.empty:
                    st.warning(f"Date column not found in equipment stats. Available columns: {', '.join(equipment_stats.columns.tolist())}")
                
                # Health Score Trend
                st.subheader("System Health Score")
                
                if not daily_metrics.empty and 'daily_health_score' in daily_metrics.columns and 'reading_date' in daily_metrics.columns:
                    fig_health = go.Figure()
                    fig_health.add_trace(go.Scatter(
                        x=daily_metrics['reading_date'],
                        y=daily_metrics['daily_health_score'],
                        name='Health Score',
                        line=dict(color='green', width=3),
                        fill='tonexty',
                        fillcolor='rgba(0,255,0,0.1)'
                    ))
                    fig_health.add_hline(y=75, line_dash="dash", line_color="orange", 
                                        annotation_text="Target: 75%")
                    
                    fig_health.update_layout(
                        title="Daily System Health Score Trend",
                        xaxis_title="Date",
                        yaxis_title="Health Score (%)",
                        yaxis=dict(range=[0, 100]),
                        hovermode='x unified',
                        height=300
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
                elif not daily_metrics.empty:
                    if 'reading_date' not in daily_metrics.columns:
                        st.warning(f"Date column not found. Available columns: {', '.join(daily_metrics.columns.tolist())}")
                    else:
                        st.info("Health score data not available in the semantic layer.")
                
                # Data Tables
                with st.expander("View Daily Metrics Data"):
                    st.dataframe(daily_metrics, use_container_width=True)
                    
                    csv = daily_metrics.to_csv(index=False)
                    st.download_button(
                        label="Download Daily Metrics CSV",
                        data=csv,
                        file_name=f"daily_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    except ImportError as e:
        st.error(f"Error importing BI query functions: {e}. Please ensure all dependencies are installed.")
    except Exception as e:
        st.error(f"Error loading BI dashboard: {e}")
