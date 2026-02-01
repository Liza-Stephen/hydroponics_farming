"""Silver Layer: Cleaned and validated data"""
from pyspark.sql.functions import col, when, upper, trim, current_timestamp, row_number, desc, regexp_replace
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.window import Window
from config.databricks_config import get_spark_session


def _to_numeric(col_name, dtype=DoubleType):
    """Convert string column to numeric, handling empty strings"""
    return when((col(col_name) == "") | col(col_name).isNull(), None) \
        .otherwise(regexp_replace(col(col_name), '"', '').cast(dtype()))


def _to_bool(col_name):
    """Convert ON/OFF string to boolean"""
    return when(upper(trim(col(col_name))) == "ON", True) \
        .when(upper(trim(col(col_name))) == "OFF", False) \
        .otherwise(None)


def clean_and_validate_data(spark, bronze_table_name, silver_table_name, s3_bronze_path, s3_silver_path):
    """
    Transform bronze data to silver layer with cleaning and validation
    
    Args:
        spark: SparkSession
        bronze_table_name: Full bronze table name (catalog.schema.table)
        silver_table_name: Full silver table name (catalog.schema.table)
        s3_bronze_path: S3 path for bronze parquet files
        s3_silver_path: S3 path for silver parquet files
    """
    print("Processing data from bronze to silver layer...")
    
    # Read from bronze parquet files in S3 (preferred) or fallback to table
    bronze_parquet_path = f"{s3_bronze_path}/parquet/iot_data"
    try:
        print(f"Reading from S3: {bronze_parquet_path}")
        df = spark.read.parquet(bronze_parquet_path)
    except Exception as e:
        print(f"Could not read from S3, falling back to table: {e}")
        df = spark.table(bronze_table_name)
    
    from pyspark.sql.functions import to_timestamp
    
    # Data cleaning and type conversion
    df_cleaned = df.select(
        col("id").cast(IntegerType()).alias("id"),
        to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss").alias("timestamp"),
        _to_numeric("pH").alias("pH"),
        _to_numeric("TDS").alias("TDS"),
        _to_numeric("water_level", IntegerType).alias("water_level"),
        _to_numeric("DHT_temp").alias("DHT_temp"),
        _to_numeric("DHT_humidity").alias("DHT_humidity"),
        _to_numeric("water_temp").alias("water_temp"),
        _to_bool("pH_reducer").alias("pH_reducer"),
        _to_bool("add_water").alias("add_water"),
        _to_bool("nutrients_adder").alias("nutrients_adder"),
        _to_bool("humidifier").alias("humidifier"),
        _to_bool("ex_fan").alias("ex_fan"),
        col("ingestion_timestamp"),
        col("source_file")
    )
    
    # Data quality checks - filter out invalid records
    df_validated = df_cleaned.filter(
        col("id").isNotNull() &
        col("timestamp").isNotNull() &
        # pH should be between 0 and 14
        ((col("pH").isNull()) | ((col("pH") >= 0) & (col("pH") <= 14))) &
        # TDS should be positive
        ((col("TDS").isNull()) | (col("TDS") >= 0)) &
        # Temperature should be reasonable (0-50°C)
        ((col("DHT_temp").isNull()) | ((col("DHT_temp") >= -10) & (col("DHT_temp") <= 50))) &
        ((col("water_temp").isNull()) | ((col("water_temp") >= 0) & (col("water_temp") <= 50))) &
        # Humidity should be 0-100%
        ((col("DHT_humidity").isNull()) | ((col("DHT_humidity") >= 0) & (col("DHT_humidity") <= 100)))
    )
    
    # Deduplication - keep latest record for same id and timestamp
    window_spec = Window.partitionBy("id", "timestamp").orderBy(desc("ingestion_timestamp"))
    df_deduplicated = df_validated.withColumn(
        "row_num", row_number().over(window_spec)
    ).filter(col("row_num") == 1).drop("row_num")
    
    # Add processing metadata
    df_final = df_deduplicated.withColumn(
        "silver_processed_timestamp", current_timestamp()
    )
    
    # Write parquet to S3
    parquet_path = f"{s3_silver_path}/parquet/iot_data"
    print(f"Writing parquet files to {parquet_path}...")
    df_final.write \
        .format("parquet") \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .save(parquet_path)
    
    # Write to Databricks table
    df_final.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(silver_table_name)
    
    print(f"Successfully processed {df_final.count()} records to silver layer")
    print(f"  - Parquet files: {parquet_path}")
    print(f"  - Databricks table: {silver_table_name}")
    return df_final


def create_silver_table(spark, config, table_name="iot_data"):
    """Create silver table as managed table"""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.silver_schema}")
    table_name_full = config.get_table_name(config.silver_schema, table_name)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name_full} USING DELTA")
    return table_name_full


def run_silver_processing():
    """Run silver layer processing"""
    spark, config = get_spark_session()
    
    silver_table_name = create_silver_table(spark, config)
    bronze_table_name = config.get_table_name(config.bronze_schema, "iot_data")
    df = clean_and_validate_data(spark, bronze_table_name, silver_table_name, config.s3_bronze_path, config.s3_silver_path)
    
    print("\nSample silver data:")
    df.show(5, truncate=False)
    print(f"\nTotal records: {df.count()}")
    return df


if __name__ == "__main__":
    # Parameters: [DATABRICKS_CATALOG, S3_BUCKET] - config reads from sys.argv
    run_silver_processing()
