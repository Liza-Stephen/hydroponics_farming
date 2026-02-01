"""Bronze Layer: Batch data ingestion"""
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


def ingest_raw_data(spark, source_path, bronze_table_name, s3_bronze_path):
    """
    Ingest raw CSV data into bronze layer
    
    Args:
        spark: SparkSession
        source_path: Path to source CSV file (S3 path)
        bronze_table_name: Full table name (catalog.schema.table)
        s3_bronze_path: S3 path for bronze parquet files
    """
    print(f"Ingesting data from {source_path} to bronze layer...")
    
    # Read CSV from S3 with schema
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
    
    # Write parquet to S3
    parquet_path = f"{s3_bronze_path}/parquet/iot_data"
    print(f"Writing parquet files to {parquet_path}...")
    df.write \
      .format("parquet") \
      .mode("append") \
      .option("compression", "snappy") \
      .save(parquet_path)
    
    # Write to Databricks table (append mode for incremental loads)
    df.write \
      .format("delta") \
      .mode("append") \
      .option("mergeSchema", "true") \
      .saveAsTable(bronze_table_name)
    
    print(f"Successfully ingested {df.count()} records to bronze layer")
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


def run_bronze_ingestion(source_csv_path=None):
    """Run bronze layer ingestion"""
    spark, config = get_spark_session()
    
    if source_csv_path:
        source_path = source_csv_path
    elif config.source_data_path:
        source_path = config.source_data_path
    else:
        raise ValueError("SOURCE_FILE_KEY is required. Set it as job parameter (sys.argv[2]). The full S3 path will be constructed as s3://{S3_BUCKET}/{SOURCE_FILE_KEY}.")
    
    bronze_table_name = create_bronze_table(spark, config)
    df = ingest_raw_data(spark, source_path, bronze_table_name, config.s3_bronze_path)
    
    print("\nSample bronze data:")
    df.show(5, truncate=False)
    return df


if __name__ == "__main__":
    # Parameters come from job configuration: [DATABRICKS_CATALOG, SOURCE_DATA_PATH]
    # Config reads them automatically via sys.argv
    run_bronze_ingestion()
