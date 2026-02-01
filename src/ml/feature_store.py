"""
Feature Store using Delta Tables

Creates and manages features from the Gold layer for ML consumption using Delta tables in Unity Catalog.
"""
from pyspark.sql.functions import col, lag, window, avg, max as spark_max, min as spark_min
from pyspark.sql.window import Window
from config.databricks_config import get_spark_session


class FeatureStoreManager:
    """Manages feature tables using Delta tables in Unity Catalog"""
    
    def __init__(self, catalog, database="feature_store"):
        """
        Initialize Feature Store Manager
        
        Args:
            catalog: Unity Catalog name
            database: Feature Store database name (default: feature_store)
        """
        self.spark, self.config = get_spark_session()
        self.catalog = catalog
        self.database = database
        
        # Create database if it doesn't exist
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.{database}")
        print(f"Feature Store database: {catalog}.{database}")
    
    def create_sensor_features(self, gold_table_name, feature_table_name):
        """
        Create time-series features from Gold layer fact table
        
        Features include:
        - Raw sensor readings (pH, TDS, temperatures, humidity)
        - Rolling statistics (moving averages, min/max)
        - Lag features (previous values)
        - Equipment state features
        - Optimal condition indicators
        
        Args:
            gold_table_name: Full table name (catalog.schema.table) for Gold fact table
            feature_table_name: Feature table name (will be created in feature_store database)
        """
        print(f"Creating sensor features from {gold_table_name}...")
        
        # Read from Gold layer
        df = self.spark.table(f"{self.catalog}.{gold_table_name}")
        
        # Define window for rolling statistics using ROWS (row-based, not time-based)
        # Assuming ~1 reading per minute: 1h ≈ 60 rows, 6h ≈ 360 rows, 24h ≈ 1440 rows
        # Adjust these values based on your actual data frequency
        rows_1h = 60   # Approximate rows for 1 hour
        rows_6h = 360  # Approximate rows for 6 hours
        rows_24h = 1440  # Approximate rows for 24 hours
        
        window_1h = Window.partitionBy().orderBy(col("timestamp_key")).rowsBetween(-rows_1h, 0)
        window_6h = Window.partitionBy().orderBy(col("timestamp_key")).rowsBetween(-rows_6h, 0)
        window_24h = Window.partitionBy().orderBy(col("timestamp_key")).rowsBetween(-rows_24h, 0)
        
        # Create features
        df_features = df.select(
            # Primary keys
            col("reading_id").alias("reading_id"),
            col("timestamp_key").alias("timestamp"),
            
            # Raw sensor readings
            col("ph_level"),
            col("tds_level"),
            col("water_level"),
            col("air_temperature"),
            col("air_humidity"),
            col("water_temperature"),
            
            # Equipment states (boolean)
            col("is_ph_reducer_on").cast("int").alias("ph_reducer_state"),
            col("is_add_water_on").cast("int").alias("add_water_state"),
            col("is_nutrients_adder_on").cast("int").alias("nutrients_adder_state"),
            col("is_humidifier_on").cast("int").alias("humidifier_state"),
            col("is_ex_fan_on").cast("int").alias("ex_fan_state"),
            
            # Optimal condition indicators
            col("is_ph_optimal").cast("int").alias("is_ph_optimal"),
            col("is_tds_optimal").cast("int").alias("is_tds_optimal"),
            col("is_temp_optimal").cast("int").alias("is_temp_optimal"),
            col("is_humidity_optimal").cast("int").alias("is_humidity_optimal"),
            
            # Lag features (previous values)
            lag(col("ph_level"), 1).over(Window.orderBy(col("timestamp_key"))).alias("ph_lag_1"),
            lag(col("tds_level"), 1).over(Window.orderBy(col("timestamp_key"))).alias("tds_lag_1"),
            lag(col("air_temperature"), 1).over(Window.orderBy(col("timestamp_key"))).alias("temp_lag_1"),
            lag(col("air_humidity"), 1).over(Window.orderBy(col("timestamp_key"))).alias("humidity_lag_1"),
            
            # Rolling statistics - pH
            avg(col("ph_level")).over(window_1h).alias("ph_avg_1h"),
            avg(col("ph_level")).over(window_6h).alias("ph_avg_6h"),
            spark_max(col("ph_level")).over(window_1h).alias("ph_max_1h"),
            spark_min(col("ph_level")).over(window_1h).alias("ph_min_1h"),
            
            # Rolling statistics - TDS
            avg(col("tds_level")).over(window_1h).alias("tds_avg_1h"),
            avg(col("tds_level")).over(window_6h).alias("tds_avg_6h"),
            spark_max(col("tds_level")).over(window_1h).alias("tds_max_1h"),
            spark_min(col("tds_level")).over(window_1h).alias("tds_min_1h"),
            
            # Rolling statistics - Temperature
            avg(col("air_temperature")).over(window_1h).alias("temp_avg_1h"),
            avg(col("air_temperature")).over(window_6h).alias("temp_avg_6h"),
            spark_max(col("air_temperature")).over(window_1h).alias("temp_max_1h"),
            spark_min(col("air_temperature")).over(window_1h).alias("temp_min_1h"),
            
            # Rolling statistics - Humidity
            avg(col("air_humidity")).over(window_1h).alias("humidity_avg_1h"),
            avg(col("air_humidity")).over(window_6h).alias("humidity_avg_6h"),
            spark_max(col("air_humidity")).over(window_1h).alias("humidity_max_1h"),
            spark_min(col("air_humidity")).over(window_1h).alias("humidity_min_1h"),
            
            # Rolling statistics - Water temperature
            avg(col("water_temperature")).over(window_1h).alias("water_temp_avg_1h"),
            avg(col("water_temperature")).over(window_6h).alias("water_temp_avg_6h"),
        )
        
        # Create feature table as Delta table in Unity Catalog
        full_table_name = f"{self.catalog}.{self.database}.{feature_table_name}"
        
        print(f"Creating feature table as Delta table: {full_table_name}")
        df_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
        print(f"Created feature table: {full_table_name}")
        
        return full_table_name
    
    def get_feature_table(self, feature_table_name):
        """Get feature table as Spark DataFrame"""
        full_table_name = f"{self.catalog}.{self.database}.{feature_table_name}"
        return self.spark.table(full_table_name)


def create_feature_store(catalog, gold_table_name, feature_table_name="sensor_features"):
    """
    Create Feature Store from Gold layer
    
    Args:
        catalog: Unity Catalog name
        gold_table_name: Full table name for Gold fact table
        feature_table_name: Name for the feature table (default: sensor_features)
    
    Returns:
        Full feature table name
    """
    manager = FeatureStoreManager(catalog)
    return manager.create_sensor_features(gold_table_name, feature_table_name)


if __name__ == "__main__":
    import sys
    
    # Parameters: [DATABRICKS_CATALOG, S3_BUCKET, GOLD_TABLE_NAME, FEATURE_TABLE_NAME]
    if len(sys.argv) < 3:
        raise ValueError("Usage: feature_store.py <DATABRICKS_CATALOG> <S3_BUCKET> [GOLD_TABLE_NAME] [FEATURE_TABLE_NAME]")
    
    catalog = sys.argv[1]
    s3_bucket = sys.argv[2]  # Not used but required for config
    gold_table_name = sys.argv[3] if len(sys.argv) > 3 else "gold.iot_data"
    feature_table_name = sys.argv[4] if len(sys.argv) > 4 else "sensor_features"
    
    create_feature_store(catalog, gold_table_name, feature_table_name)
