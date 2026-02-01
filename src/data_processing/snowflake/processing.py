"""
Snowflake Layer: Load data from Gold layer to Snowflake

NOTE: This module runs EXCLUSIVELY in Databricks, not locally.
The snowflake-connector-python and pandas packages are pre-installed
in Databricks runtime and do not need to be installed locally.
"""
import snowflake.connector
from config.databricks_config import get_spark_session
from config.snowflake_config import SnowflakeConfig


def create_snowflake_connection(config):
    """Create Snowflake connection using password or key pair authentication"""
    print(f"Connecting to Snowflake account: {config.account}")
    
    # Build connection parameters
    conn_params = {
        "account": config.account,
        "user": config.user,
        "warehouse": config.warehouse,
        "database": config.database,
        "schema": config.schema
    }
    
    # Add authentication method
    if config.auth_method == "key_pair":
        print("Using key pair authentication")
        conn_params["private_key"] = config.private_key
        if hasattr(config, 'private_key_passphrase') and config.private_key_passphrase:
            conn_params["private_key_passphrase"] = config.private_key_passphrase
    else:
        print("Using password authentication")
        conn_params["password"] = config.password
    
    conn = snowflake.connector.connect(**conn_params)
    print(f"✓ Connected to Snowflake: {config.database}.{config.schema}")
    return conn


def create_snowflake_schema(conn, config):
    """Create Snowflake database and schema if they don't exist"""
    cursor = conn.cursor()
    
    try:
        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.database}")
        print(f"✓ Database {config.database} ready")
        
        # Use database
        cursor.execute(f"USE DATABASE {config.database}")
        
        # Create schema if not exists
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.schema}")
        print(f"✓ Schema {config.schema} ready")
        
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
    
    # Connect to Snowflake and load data
    conn = create_snowflake_connection(config)
    cursor = conn.cursor()
    
    try:
        # Create table if not exists (using pandas schema)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {snowflake_table_name} (
            {', '.join([f"{col} {_get_snowflake_type(pandas_df[col].dtype)}" for col in pandas_df.columns])}
        )
        """
        cursor.execute(create_table_sql)
        print(f"  - Table {snowflake_table_name} ready")
        
        # Truncate table before loading (overwrite mode)
        cursor.execute(f"TRUNCATE TABLE IF EXISTS {snowflake_table_name}")
        
        # Write data to Snowflake using INSERT
        # For large datasets, consider using COPY INTO or Snowflake's bulk loading
        if len(pandas_df) > 0:
            # Convert DataFrame to list of tuples
            values = [tuple(row) for row in pandas_df.values]
            placeholders = ', '.join(['?' for _ in pandas_df.columns])
            columns = ', '.join(pandas_df.columns)
            
            insert_sql = f"INSERT INTO {snowflake_table_name} ({columns}) VALUES ({placeholders})"
            cursor.executemany(insert_sql, values)
            conn.commit()
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
    
    # Connect to Snowflake and load data
    conn = create_snowflake_connection(config)
    cursor = conn.cursor()
    
    try:
        # Create table if not exists
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {snowflake_table_name} (
            {', '.join([f"{col} {_get_snowflake_type(pandas_df[col].dtype)}" for col in pandas_df.columns])}
        )
        """
        cursor.execute(create_table_sql)
        print(f"  - Table {snowflake_table_name} ready")
        
        # Truncate table before loading
        cursor.execute(f"TRUNCATE TABLE IF EXISTS {snowflake_table_name}")
        
        # Batch insert (for large tables, consider using COPY INTO with S3 staging)
        if len(pandas_df) > 0:
            # For large datasets, insert in batches
            batch_size = 10000
            values = [tuple(row) for row in pandas_df.values]
            placeholders = ', '.join(['?' for _ in pandas_df.columns])
            columns = ', '.join(pandas_df.columns)
            insert_sql = f"INSERT INTO {snowflake_table_name} ({columns}) VALUES ({placeholders})"
            
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


def _get_snowflake_type(pandas_dtype):
    """Map pandas dtype to Snowflake SQL type"""
    dtype_str = str(pandas_dtype)
    
    if 'int' in dtype_str:
        return 'INTEGER'
    elif 'float' in dtype_str:
        return 'FLOAT'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    elif 'datetime' in dtype_str:
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
