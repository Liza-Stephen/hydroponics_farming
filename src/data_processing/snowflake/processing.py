"""
Snowflake Layer: Load data from Gold layer to Snowflake

NOTE: This module runs EXCLUSIVELY in Databricks, not locally.
The snowflake-connector-python and pandas packages are pre-installed
in Databricks runtime and do not need to be installed locally.
"""
import snowflake.connector
import snowflake.connector.errors
from config.databricks_config import get_spark_session
from config.snowflake_config import SnowflakeConfig


def create_snowflake_connection(config):
    """Create Snowflake connection using password authentication"""
    print(f"Connecting to Snowflake account: {config.account}")
    print(f"Using user: {config.user}")
    
    # Build connection parameters
    conn_params = config.get_connection_params()
    
    conn = snowflake.connector.connect(**conn_params)
    print(f"✓ Connected to Snowflake: {config.database}.{config.schema}")
    return conn


def create_snowflake_schema(conn, config):
    """Create Snowflake database and schema if they don't exist"""
    cursor = conn.cursor()
    
    try:
        # Try to create database if not exists
        try:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.database}")
            print(f"✓ Database {config.database} ready")
        except snowflake.connector.errors.ProgrammingError as e:
            # Database might already exist - check if we can use it
            if "already exists" in str(e).lower():
                print(f"⚠️  Database {config.database} already exists. Attempting to use it...")
                # Try to use the database - this will fail if no privileges
                try:
                    cursor.execute(f"USE DATABASE {config.database}")
                    print(f"✓ Database {config.database} is accessible")
                except snowflake.connector.errors.ProgrammingError as use_error:
                    error_msg = str(use_error)
                    if "no privileges" in error_msg.lower() or "privileges" in error_msg.lower():
                        print(f"\n✗ ERROR: Database {config.database} exists but current role has no privileges on it.")
                        print(f"\nTo fix this, run the following SQL in Snowflake as ACCOUNTADMIN:")
                        print(f"  GRANT USAGE ON DATABASE {config.database} TO ROLE PUBLIC;")
                        print(f"  GRANT CREATE SCHEMA ON DATABASE {config.database} TO ROLE PUBLIC;")
                        print(f"  GRANT ALL PRIVILEGES ON DATABASE {config.database} TO ROLE PUBLIC;")
                        raise
                    else:
                        raise
            else:
                raise
        
        # Use database (if not already using it)
        try:
            cursor.execute(f"USE DATABASE {config.database}")
        except:
            pass  # Already using it
        
        # Create schema if not exists (database is already in use)
        try:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.schema}")
            print(f"✓ Schema {config.schema} ready")
        except snowflake.connector.errors.ProgrammingError as e:
            if "already exists" in str(e).lower():
                print(f"✓ Schema {config.schema} already exists")
            elif "no privileges" in str(e).lower() or "privileges" in str(e).lower():
                print(f"\n✗ ERROR: Cannot create schema. Current role has no privileges.")
                print(f"\nTo fix this, run the following SQL in Snowflake as ACCOUNTADMIN:")
                print(f"  GRANT CREATE SCHEMA ON DATABASE {config.database} TO ROLE PUBLIC;")
                raise
            else:
                raise
        
    finally:
        cursor.close()


def load_dimension_table(spark, config, table_name, snowflake_table_name):
    """Load dimension table from Gold layer to Snowflake"""
    print(f"\nLoading {table_name} to Snowflake...")
    
    # Read from Gold parquet files (preferred) or Gold table (fallback)
    gold_parquet_path = f"{config.s3_gold_path}/parquet/{table_name}"
    try:
        print(f"Reading from S3: {gold_parquet_path}")
        df = spark.read.parquet(gold_parquet_path)
    except Exception as e:
        print(f"Could not read from S3, falling back to Databricks table: {e}")
        gold_table_name = config.get_gold_table_name(table_name)
        df = spark.table(gold_table_name)
    
    # Convert Spark DataFrame to Pandas for Snowflake upload
    pandas_df = df.toPandas()
    print(f"  - Loaded {len(pandas_df)} records")
    
    # Convert timestamp/datetime columns to ISO format strings (Snowflake doesn't support binding timestamps directly)
    import pandas as pd
    import numpy as np
    for col in pandas_df.columns:
        if pd.api.types.is_datetime64_any_dtype(pandas_df[col]):
            # Convert to ISO format string (YYYY-MM-DD HH:MM:SS) that Snowflake can parse
            # Handle NaT/None values properly - convert to None first, then format non-null values
            pandas_df[col] = pandas_df[col].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else None
            )
    
    # Replace all remaining NaN/NaT values with None for proper SQL NULL handling
    pandas_df = pandas_df.where(pd.notna(pandas_df), None)
    
    # Connect to Snowflake and load data
    conn = create_snowflake_connection(config)
    cursor = conn.cursor()
    
    try:
        # Create table if not exists (using pandas schema)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {snowflake_table_name} (
            {', '.join([f"{col} {_get_snowflake_type(pandas_df[col].dtype, col)}" for col in pandas_df.columns])}
        )
        """
        cursor.execute(create_table_sql)
        print(f"  - Table {snowflake_table_name} ready")
        
        # Truncate table before loading (overwrite mode)
        cursor.execute(f"TRUNCATE TABLE IF EXISTS {snowflake_table_name}")
        
        # Write data to Snowflake using INSERT
        # For large datasets, consider using COPY INTO or Snowflake's bulk loading
        if len(pandas_df) > 0:
            # Convert DataFrame to list of tuples, ensuring proper NULL handling
            pandas_df_clean = pandas_df.where(pd.notnull(pandas_df), None)
            
            # Convert to list of tuples - each row becomes a tuple
            values = [tuple(row) for row in pandas_df_clean.values]
            
            # Verify we have the right number of columns
            num_cols = len(pandas_df.columns)
            placeholders = ', '.join(['%s' for _ in range(num_cols)])
            columns = ', '.join(pandas_df.columns)
            
            insert_sql = f"INSERT INTO {snowflake_table_name} ({columns}) VALUES ({placeholders})"
            
            # Insert in smaller batches to avoid issues
            batch_size = 1000
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]

                print("INSERT SQL:", insert_sql)
                print("Placeholders:", insert_sql.count("%s"))
                print("Row length:", len(batch[0]))
                print("Sample row:", batch[0])
                
                cursor.executemany(insert_sql, batch)
                conn.commit()
                if len(values) > batch_size:
                    print(f"  - Inserted batch {i//batch_size + 1} ({len(batch)} records)")
            
            print(f"  - Inserted {len(pandas_df)} records into {snowflake_table_name}")
        
    finally:
        cursor.close()
        conn.close()
    
    return snowflake_table_name


def load_fact_table(spark, config, table_name, snowflake_table_name):
    """Load fact table from Gold layer to Snowflake"""
    print(f"\nLoading {table_name} to Snowflake...")
    
    # Read from Gold parquet files (preferred) or Gold table (fallback)
    gold_parquet_path = f"{config.s3_gold_path}/parquet/{table_name}"
    try:
        print(f"Reading from S3: {gold_parquet_path}")
        df = spark.read.parquet(gold_parquet_path)
    except Exception as e:
        print(f"Could not read from S3, falling back to Databricks table: {e}")
        gold_table_name = config.get_gold_table_name(table_name)
        df = spark.table(gold_table_name)
    
    # For large fact tables, use COPY INTO for better performance
    # For now, we'll use batch inserts similar to dimensions
    pandas_df = df.toPandas()
    print(f"  - Loaded {len(pandas_df)} records")
    
    # Convert timestamp/datetime columns to ISO format strings (Snowflake doesn't support binding timestamps directly)
    import pandas as pd
    for col in pandas_df.columns:
        if pd.api.types.is_datetime64_any_dtype(pandas_df[col]):
            # Convert to ISO format string (YYYY-MM-DD HH:MM:SS) that Snowflake can parse
            pandas_df[col] = pandas_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Replace 'NaT' (pandas null timestamp) with None
            pandas_df[col] = pandas_df[col].replace('NaT', None)
    
    # Connect to Snowflake and load data
    conn = create_snowflake_connection(config)
    cursor = conn.cursor()
    
    try:
        # Create table if not exists
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {snowflake_table_name} (
            {', '.join([f"{col} {_get_snowflake_type(pandas_df[col].dtype, col)}" for col in pandas_df.columns])}
        )
        """
        cursor.execute(create_table_sql)
        print(f"  - Table {snowflake_table_name} ready")
        
        # Truncate table before loading
        cursor.execute(f"TRUNCATE TABLE IF EXISTS {snowflake_table_name}")
        
        # Batch insert (for large tables, consider using COPY INTO with S3 staging)
        if len(pandas_df) > 0:
            # Convert DataFrame to list of tuples, ensuring proper NULL handling
            pandas_df_clean = pandas_df.where(pd.notnull(pandas_df), None)
            
            # Convert to list of tuples - each row becomes a tuple
            values = [tuple(row) for row in pandas_df_clean.values]
            
            # Verify we have the right number of columns
            num_cols = len(pandas_df.columns)
            placeholders = ', '.join(['?' for _ in range(num_cols)])
            columns = ', '.join(pandas_df.columns)
            insert_sql = f"INSERT INTO {snowflake_table_name} ({columns}) VALUES ({placeholders})"
            
            # For large datasets, insert in batches
            batch_size = 10000
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                cursor.executemany(insert_sql, batch)
                conn.commit()
                print(f"  - Inserted batch {i//batch_size + 1} ({len(batch)} records)")
            
            print(f"  - Total {len(pandas_df)} records inserted into {snowflake_table_name}")
        
    finally:
        cursor.close()
        conn.close()
    
    return snowflake_table_name


def _get_snowflake_type(pandas_dtype, column_name=None):
    """Map pandas dtype to Snowflake SQL type"""
    dtype_str = str(pandas_dtype)
    
    if 'int' in dtype_str:
        return 'INTEGER'
    elif 'float' in dtype_str:
        return 'FLOAT'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    elif dtype_str == 'object':
        # Check if column name suggests timestamp (after string conversion)
        # Snowflake can auto-convert ISO format strings to TIMESTAMP_NTZ
        if column_name and ('timestamp' in column_name.lower() or 
                           (column_name.lower().endswith('_time') or 
                            column_name.lower() == 'timestamp_key' or
                            column_name.lower() == 'timestamp')):
            return 'TIMESTAMP_NTZ'  # Snowflake will auto-convert ISO string to timestamp
        else:
            return 'VARCHAR(16777216)'  # For other strings
    elif 'datetime' in dtype_str:
        # This shouldn't happen after conversion, but handle it just in case
        return 'TIMESTAMP_NTZ'
    else:
        return 'VARCHAR(16777216)'  # Default to large VARCHAR for strings


def load_gold_to_snowflake(spark, config):
    """Load all Gold layer tables to Snowflake"""
    print("="*60)
    print("LOADING GOLD LAYER TO SNOWFLAKE")
    print("="*60)
    
    # Create Snowflake connection and schema
    conn = create_snowflake_connection(config)
    create_snowflake_schema(conn, config)
    conn.close()
    
    # Load dimension tables
    dim_time_table = config.get_snowflake_table_name("dim_time")
    load_dimension_table(spark, config, "dim_time", dim_time_table)
    
    dim_equipment_table = config.get_snowflake_table_name("dim_equipment")
    load_dimension_table(spark, config, "dim_equipment", dim_equipment_table)
    
    # Load fact table
    fact_table = config.get_snowflake_table_name("iot_data")
    load_fact_table(spark, config, "iot_data", fact_table)
    
    print("\n✓ All Gold layer tables loaded to Snowflake!")
    
    return {
        "dim_time": dim_time_table,
        "dim_equipment": dim_equipment_table,
        "iot_data": fact_table
    }


def run_snowflake_processing():
    """Run Snowflake layer processing"""
    spark, databricks_config = get_spark_session()
    snowflake_config = SnowflakeConfig()
    
    tables = load_gold_to_snowflake(spark, snowflake_config)
    
    # Verify data in Snowflake
    conn = create_snowflake_connection(snowflake_config)
    cursor = conn.cursor()
    
    try:
        for table_name, full_table_name in tables.items():
            cursor.execute(f"SELECT COUNT(*) FROM {full_table_name}")
            count = cursor.fetchone()[0]
            print(f"\n✓ {table_name}: {count} records in Snowflake")
    finally:
        cursor.close()
        conn.close()
    
    return tables


if __name__ == "__main__":
    # Parameters: [DATABRICKS_CATALOG, S3_BUCKET, SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, 
    #              SNOWFLAKE_PASSWORD, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA]
    # Config reads them automatically via sys.argv
    run_snowflake_processing()
