"""Bronze Layer: API-based data ingestion"""
import sys
from pyspark.sql.functions import current_timestamp, lit, explode, col as spark_col
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


def ingest_api_data(spark, json_dir, bronze_table_name, s3_bronze_path):
    """
    Ingest data from JSON files (from API) into bronze layer
    
    Args:
        spark: SparkSession
        json_dir: S3 path containing JSON files from API
        bronze_table_name: Full table name (catalog.schema.table)
        s3_bronze_path: S3 path for bronze parquet files
    """
    print(f"Ingesting API data from {json_dir} to bronze layer...")
    
    # Read JSON files from S3
    df = spark.read \
        .option("multiline", "true") \
        .json(f"{json_dir}/*.json")
    
    # Handle both single records and batch records
    # If records column exists, explode it; otherwise use the root level
    if "records" in df.columns:
        df = df.select(explode(spark_col("records")).alias("record")).select("record.*")
    
    # Select and normalize columns to match schema
    df_normalized = df.select(
        spark_col("id").cast("string").alias("id"),
        spark_col("timestamp").cast("string").alias("timestamp"),
        spark_col("pH").cast("string").alias("pH"),
        spark_col("TDS").cast("string").alias("TDS"),
        spark_col("water_level").cast("string").alias("water_level"),
        spark_col("DHT_temp").cast("string").alias("DHT_temp"),
        spark_col("DHT_humidity").cast("string").alias("DHT_humidity"),
        spark_col("water_temp").cast("string").alias("water_temp"),
        spark_col("pH_reducer").cast("string").alias("pH_reducer"),
        spark_col("add_water").cast("string").alias("add_water"),
        spark_col("nutrients_adder").cast("string").alias("nutrients_adder"),
        spark_col("humidifier").cast("string").alias("humidifier"),
        spark_col("ex_fan").cast("string").alias("ex_fan"),
    )
    
    # Apply schema to ensure consistency
    schema = create_bronze_schema()
    df = spark.createDataFrame(df_normalized.rdd, schema=schema)
    
    # Add metadata columns
    df = df.withColumn("ingestion_timestamp", current_timestamp()) \
           .withColumn("source_file", lit("api_ingestion"))
    
    # Write parquet to S3
    parquet_path = f"{s3_bronze_path}/parquet/iot_data"
    print(f"Writing parquet files to {parquet_path}...")
    df.write \
      .format("parquet") \
      .mode("append") \
      .option("compression", "snappy") \
      .save(parquet_path)
    
    # Write to Databricks table (append mode)
    df.write \
      .format("delta") \
      .mode("append") \
      .option("mergeSchema", "true") \
      .saveAsTable(bronze_table_name)
    
    print(f"Successfully ingested {df.count()} records from API to bronze layer")
    print(f"  - Parquet files: {parquet_path}")
    print(f"  - Databricks table: {bronze_table_name}")
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
    
    # Read JSON directory file key from job parameters (sys.argv)
    # Job parameters: sys.argv[1] = DATABRICKS_CATALOG, sys.argv[2] = S3_BUCKET, sys.argv[3] = JSON_DIR_FILE_KEY
    # Construct full S3 path: s3://{S3_BUCKET}/{JSON_DIR_FILE_KEY}
    if json_dir_path is None:
        if len(sys.argv) > 3:
            # Bronze task with all parameters: sys.argv[3] = JSON_DIR_FILE_KEY
            file_key = sys.argv[3]
            # Remove leading slash if present
            if file_key.startswith("/"):
                file_key = file_key[1:]
            json_dir_path = f"s3://{config.s3_bucket}/{file_key}"
        else:
            raise ValueError("JSON_DIR_FILE_KEY is required. Set it as job parameter (sys.argv[3]). The full S3 path will be constructed as s3://{S3_BUCKET}/{JSON_DIR_FILE_KEY}.")
    
    bronze_table_name = create_bronze_table(spark, config)
    df = ingest_api_data(spark, json_dir_path, bronze_table_name, config.s3_bronze_path)
    
    if df:
        print("\nSample API ingested data:")
        df.show(5, truncate=False)
    
    return df


if __name__ == "__main__":
    # Parameters come from job configuration: [DATABRICKS_CATALOG, S3_BUCKET, JSON_DIR_FILE_KEY]
    # Config reads them automatically via sys.argv
    run_api_ingestion()
