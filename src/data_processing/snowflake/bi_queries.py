"""
BI Dashboard Queries for Semantic Layer
Queries dbt models from Snowflake for BI dashboard
"""
import os
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector

# Load environment variables from .env file
# bi_queries.py is at: src/data_processing/snowflake/bi_queries.py
# .env is at project root, so go up 4 levels
if __file__:
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
else:
    # Fallback if __file__ is not available (e.g., in some execution contexts)
    env_file = Path.cwd() / ".env"

load_dotenv(env_file, override=False)  # override=False to not overwrite existing env vars



def get_snowflake_connection() -> Optional[snowflake.connector.SnowflakeConnection]:
    """Create Snowflake connection from environment variables"""
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
    database = os.environ.get("SNOWFLAKE_DATABASE")
    schema = os.environ.get("SNOWFLAKE_SCHEMA")
    
    if not all([account, user, password, warehouse, database, schema]):
        return None
    
    try:
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        return conn
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None


def query_semantic_layer(query: str) -> Optional[pd.DataFrame]:
    """Execute SQL query against Snowflake semantic layer using direct connector"""
    conn = get_snowflake_connection()
    if conn is None:
        print("Error: Could not create Snowflake connection. Check credentials.")
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        # Normalize column names to lowercase (Snowflake returns uppercase)
        df.columns = df.columns.str.lower()
        cursor.close()
        conn.close()
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        print(f"Query: {query[:200]}...")  # Print first 200 chars of query for debugging
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        try:
            conn.close()
        except:
            pass
        return None


def get_daily_metrics(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Get daily aggregated metrics from semantic layer
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, uses MIN(reading_date)
        end_date: End date in 'YYYY-MM-DD' format. If None, uses MAX(reading_date)
    """
    database = os.environ.get('SNOWFLAKE_DATABASE', 'HYDROPONICS_DB')
    schema = os.environ.get('SNOWFLAKE_SCHEMA', 'ANALYTICS')
    
    # Build WHERE clause
    if start_date and end_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= '{end_date}'"
    elif start_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{schema}.daily_environment_metrics)"
    elif end_date:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{schema}.daily_environment_metrics) AND reading_date <= '{end_date}'"
    else:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{schema}.daily_environment_metrics) AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{schema}.daily_environment_metrics)"
    
    query = f"""
    SELECT 
        reading_date,
        daily_avg_ph,
        daily_avg_tds,
        daily_avg_air_temp,
        daily_avg_air_humidity,
        daily_avg_water_temp,
        daily_avg_water_level,
        daily_min_ph,
        daily_max_ph,
        daily_min_tds,
        daily_max_tds,
        daily_min_air_temp,
        daily_max_air_temp,
        total_readings,
        total_ph_optimal,
        total_tds_optimal,
        total_temp_optimal,
        total_humidity_optimal,
        total_environment_optimal,
        total_ph_reducer_activations,
        total_water_additions,
        total_nutrient_additions,
        total_humidifier_activations,
        total_fan_activations,
        avg_ph_optimality_pct,
        avg_tds_optimality_pct,
        avg_temp_optimality_pct,
        avg_humidity_optimality_pct,
        avg_environment_optimality_pct,
        daily_health_score,
        daily_environment_efficiency_pct
    FROM {database}.{schema}.daily_environment_metrics
    {where_clause}
    ORDER BY reading_date DESC
    """
    return query_semantic_layer(query)


def get_hourly_metrics(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Get hourly aggregated metrics from semantic layer
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, uses MIN(reading_date)
        end_date: End date in 'YYYY-MM-DD' format. If None, uses MAX(reading_date)
    """
    database = os.environ.get('SNOWFLAKE_DATABASE', 'HYDROPONICS_DB')
    intermediate_schema = os.environ.get('SNOWFLAKE_INTERMEDIATE_SCHEMA')
    
    # Build WHERE clause
    if start_date and end_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= '{end_date}'"
    elif start_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{intermediate_schema}.int_sensor_metrics)"
    elif end_date:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{intermediate_schema}.int_sensor_metrics) AND reading_date <= '{end_date}'"
    else:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{intermediate_schema}.int_sensor_metrics) AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{intermediate_schema}.int_sensor_metrics)"
    
    query = f"""
    SELECT 
        reading_date,
        reading_hour,
        avg_ph,
        avg_tds,
        avg_air_temp,
        avg_air_humidity,
        avg_water_temp,
        avg_water_level,
        min_ph,
        max_ph,
        min_tds,
        max_tds,
        min_air_temp,
        max_air_temp,
        reading_count,
        ph_optimal_count,
        tds_optimal_count,
        temp_optimal_count,
        humidity_optimal_count,
        environment_optimal_count,
        ph_reducer_activations,
        water_additions,
        nutrient_additions,
        humidifier_activations,
        fan_activations,
        ph_optimality_pct,
        tds_optimality_pct,
        temp_optimality_pct,
        humidity_optimality_pct,
        environment_optimality_pct
    FROM {database}.{intermediate_schema}.int_sensor_metrics
    {where_clause}
    ORDER BY reading_date DESC, reading_hour DESC
    """
    return query_semantic_layer(query)


def get_sensor_readings(limit: int = 1000) -> Optional[pd.DataFrame]:
    """Get recent sensor readings from fact table (all available data)"""
    database = os.environ.get('SNOWFLAKE_DATABASE', 'HYDROPONICS_DB')
    schema = os.environ.get('SNOWFLAKE_SCHEMA', 'ANALYTICS')
    query = f"""
    SELECT 
        reading_id,
        timestamp_key,
        reading_date,
        reading_hour,
        day_name,
        month_name,
        day_type,
        ph_value,
        tds_value,
        air_temp_celsius,
        air_humidity_pct,
        water_temp_celsius,
        water_level_pct,
        ph_reducer_on,
        add_water_on,
        nutrients_adder_on,
        humidifier_on,
        ex_fan_on,
        is_ph_optimal,
        is_tds_optimal,
        is_temp_optimal,
        is_humidity_optimal,
        is_environment_optimal,
        ph_status,
        tds_status,
        temperature_status
    FROM {database}.{schema}.fct_sensor_readings
    WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{schema}.fct_sensor_readings)
        AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{schema}.fct_sensor_readings)
    ORDER BY timestamp_key DESC
    LIMIT {limit}
    """
    return query_semantic_layer(query)


def get_equipment_usage_stats(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Get equipment usage statistics
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, uses MIN(reading_date)
        end_date: End date in 'YYYY-MM-DD' format. If None, uses MAX(reading_date)
    """
    database = os.environ.get('SNOWFLAKE_DATABASE', 'HYDROPONICS_DB')
    schema = os.environ.get('SNOWFLAKE_SCHEMA', 'ANALYTICS')
    
    # Build WHERE clause
    if start_date and end_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= '{end_date}'"
    elif start_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{schema}.daily_environment_metrics)"
    elif end_date:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{schema}.daily_environment_metrics) AND reading_date <= '{end_date}'"
    else:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{schema}.daily_environment_metrics) AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{schema}.daily_environment_metrics)"
    
    query = f"""
    SELECT 
        reading_date,
        total_ph_reducer_activations as ph_reducer,
        total_water_additions as water_addition,
        total_nutrient_additions as nutrients,
        total_humidifier_activations as humidifier,
        total_fan_activations as exhaust_fan,
        total_readings
    FROM {database}.{schema}.daily_environment_metrics
    {where_clause}
    ORDER BY reading_date DESC
    """
    return query_semantic_layer(query)


def get_optimality_trends(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Get optimality trends over time
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, uses MIN(reading_date)
        end_date: End date in 'YYYY-MM-DD' format. If None, uses MAX(reading_date)
    """
    database = os.environ.get('SNOWFLAKE_DATABASE', 'HYDROPONICS_DB')
    schema = os.environ.get('SNOWFLAKE_SCHEMA', 'ANALYTICS')
    
    # Build WHERE clause
    if start_date and end_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= '{end_date}'"
    elif start_date:
        where_clause = f"WHERE reading_date >= '{start_date}' AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{schema}.daily_environment_metrics)"
    elif end_date:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{schema}.daily_environment_metrics) AND reading_date <= '{end_date}'"
    else:
        where_clause = f"WHERE reading_date >= (SELECT MIN(reading_date) FROM {database}.{schema}.daily_environment_metrics) AND reading_date <= (SELECT MAX(reading_date) FROM {database}.{schema}.daily_environment_metrics)"
    
    query = f"""
    SELECT 
        reading_date,
        avg_ph_optimality_pct as ph_optimality,
        avg_tds_optimality_pct as tds_optimality,
        avg_temp_optimality_pct as temp_optimality,
        avg_humidity_optimality_pct as humidity_optimality,
        avg_environment_optimality_pct as environment_optimality,
        daily_health_score
    FROM {database}.{schema}.daily_environment_metrics
    {where_clause}
    ORDER BY reading_date DESC
    """
    return query_semantic_layer(query)
