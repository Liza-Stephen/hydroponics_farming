"""Bronze Layer: API-based data ingestion"""
import json
import os
import sys
from pathlib import Path
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType
from config.databricks_config import get_spark_session


def create_bronze_schema():
    """Define schema for bronze layer"""
    return StructType([
        StructField("id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("pH", StringType(), True),
        StructField("TDS", StringType(), True),
        StructField("water_level", StringType(), True),
        StructField("DHT_temp", StringType(), True),
        StructField("DHT_humidity", StringType(), True),
        StructField("water_temp", StringType(), True),
        StructField("pH_reducer", StringType(), True),
        StructField("add_water", StringType(), True),
        StructField("nutrients_adder", StringType(), True),
        StructField("humidifier", StringType(), True),
        StructField("ex_fan", StringType(), True),
    ])


def ingest_api_data(spark, json_dir, bronze_table_name):
    """
    Ingest data from JSON files (from API) into bronze layer
    
    Args:
        spark: SparkSession
        json_dir: Directory containing JSON files from API
        bronze_table_name: Full table name (catalog.schema.table)
    """
    print(f"Ingesting API data from {json_dir} to bronze layer...")
    
    # Read JSON files
    json_files = list(Path(json_dir).glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return None
    
    print(f"Found {len(json_files)} JSON files in {json_dir}")
    
    # Read all JSON files and collect records
    all_records = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Handle both single records and batch records
            if "records" in data:
                records = data["records"]
            else:
                records = [data]
            
            # Convert to list of dictionaries with proper field names
            for record in records:
                normalized = {
                    "id": str(record.get("id", "")),
                    "timestamp": str(record.get("timestamp", "")),
                    "pH": str(record.get("pH", "")),
                    "TDS": str(record.get("TDS", "")),
                    "water_level": str(record.get("water_level", "")),
                    "DHT_temp": str(record.get("DHT_temp", "")),
                    "DHT_humidity": str(record.get("DHT_humidity", "")),
                    "water_temp": str(record.get("water_temp", "")),
                    "pH_reducer": str(record.get("pH_reducer", "")),
                    "add_water": str(record.get("add_water", "")),
                    "nutrients_adder": str(record.get("nutrients_adder", "")),
                    "humidifier": str(record.get("humidifier", "")),
                    "ex_fan": str(record.get("ex_fan", "")),
                }
                all_records.append(normalized)
        
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    if not all_records:
        print("No data to ingest")
        return None
    
    # Create DataFrame from all records
    schema = create_bronze_schema()
    df = spark.createDataFrame(all_records, schema=schema)
    
    # Add metadata columns
    df = df.withColumn("ingestion_timestamp", current_timestamp()) \
           .withColumn("source_file", lit("api_ingestion"))
    
    # Write to bronze table (append mode)
    df.write \
      .format("delta") \
      .mode("append") \
      .option("mergeSchema", "true") \
      .saveAsTable(bronze_table_name)
    
    print(f"Successfully ingested {df.count()} records from API to bronze layer")
    return df


def create_bronze_table(spark, config, table_name="iot_data"):
    """Create bronze table as managed table"""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.bronze_schema}")
    table_name_full = config.get_table_name(config.bronze_schema, table_name)
    # Create managed table (Unity Catalog manages storage)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name_full} USING DELTA")
    return table_name_full


def run_api_ingestion(json_dir_path=None):
    """Run API-based bronze layer ingestion"""
    spark, config = get_spark_session()
    
    # Read JSON directory from job parameters (sys.argv) first, then fall back to argument or environment
    # Job parameters: sys.argv[1] = DATABRICKS_CATALOG, sys.argv[2] = JSON_DIR_PATH
    if json_dir_path is None:
        if len(sys.argv) > 2:
            json_dir_path = sys.argv[2]
        else:
            json_dir_path = os.getenv("JSON_DIR_PATH", "api_data")
    
    bronze_table_name = create_bronze_table(spark, config)
    df = ingest_api_data(spark, json_dir_path, bronze_table_name)
    
    if df:
        print("\nSample API ingested data:")
        df.show(5, truncate=False)
    
    return df


if __name__ == "__main__":
    # Parameters come from job configuration: [DATABRICKS_CATALOG, JSON_DIR_PATH]
    # Config reads them automatically via sys.argv
    run_api_ingestion()
