"""Bronze Layer: Raw data ingestion"""
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


def ingest_raw_data(spark, source_path, bronze_table_path):
    """
    Ingest raw CSV data into bronze layer
    
    Args:
        spark: SparkSession
        source_path: Path to source CSV file
        bronze_table_path: Path where bronze Delta table will be stored
    """
    print(f"Ingesting data from {source_path} to bronze layer...")
    
    # Read CSV with schema
    schema = create_bronze_schema()
    df = spark.read \
        .option("header", "true") \
        .option("quote", '"') \
        .option("escape", '"') \
        .schema(schema) \
        .csv(source_path)
    
    # Add metadata columns
    df = df.withColumn("ingestion_timestamp", current_timestamp()) \
           .withColumn("source_file", lit(source_path.split("/")[-1]))
    
    # Write to Delta table (append mode for incremental loads)
    df.write \
      .format("delta") \
      .mode("append") \
      .option("mergeSchema", "true") \
      .save(bronze_table_path)
    
    print(f"Successfully ingested {df.count()} records to bronze layer")
    return df


def create_bronze_table(spark, config, table_name="iot_data"):
    """Create bronze table"""
    bronze_path = f"{config.bronze_path}/{table_name}"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.bronze_schema}")
    table_name_full = config.get_table_name(config.bronze_schema, table_name)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name_full} USING DELTA LOCATION '{bronze_path}'")
    return bronze_path


def run_bronze_ingestion(source_csv_path=None):
    """Run bronze layer ingestion"""
    spark, config = get_spark_session()
    source_path = source_csv_path or config.source_data_path
    
    bronze_path = create_bronze_table(spark, config)
    df = ingest_raw_data(spark, source_path, bronze_path)
    
    print("\nSample bronze data:")
    df.show(5, truncate=False)
    return df
