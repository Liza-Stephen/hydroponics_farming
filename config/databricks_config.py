"""Databricks configuration"""
import os
import sys
from pyspark.sql import SparkSession

# Load environment variables if not in Databricks
if not os.getenv("DATABRICKS_RUNTIME_VERSION"):
    from dotenv import load_dotenv
    load_dotenv()


class DatabricksConfig:
    """Configuration for Databricks"""
    
    def __init__(self):
        # Read from job parameters (sys.argv) first, then fall back to environment variables
        # Job parameters: sys.argv[1] = DATABRICKS_CATALOG, sys.argv[2] = SOURCE_DATA_PATH
        if len(sys.argv) > 1:
            self.catalog = sys.argv[1]
        else:
            self.catalog = os.getenv("DATABRICKS_CATALOG")
        
        if not self.catalog:
            raise ValueError("DATABRICKS_CATALOG is required. Set it as job parameter or environment variable.")
        
        if self.catalog in ["spark_catalog", "hive_metastore"]:
            raise ValueError(f"Invalid catalog '{self.catalog}'. Use Unity Catalog (e.g., 'main').")
        
        # Read SOURCE_DATA_PATH from parameters or environment
        if len(sys.argv) > 2:
            self.source_data_path = sys.argv[2]
        else:
            self.source_data_path = os.getenv("SOURCE_DATA_PATH", f"/Volumes/{self.catalog}/bronze/raw_data/iot_data_raw.csv")
        
        self.bronze_schema = "bronze"
        self.silver_schema = "silver"
        self.gold_schema = "gold"
        self.bronze_path = f"/Volumes/{self.catalog}/{self.bronze_schema}"
        self.silver_path = f"/Volumes/{self.catalog}/{self.silver_schema}"
        self.gold_path = f"/Volumes/{self.catalog}/{self.gold_schema}"
    
    def get_table_name(self, schema, table):
        """Get full table name: catalog.schema.table"""
        return f"{self.catalog}.{schema}.{table}"


def get_spark_session():
    """Get Spark session and set Unity Catalog"""
    config = DatabricksConfig()
    
    spark = SparkSession.getActiveSession() or SparkSession.builder \
        .appName("HydroponicsDataProcessing") \
        .getOrCreate()
    
    spark.sql(f"USE CATALOG {config.catalog}")
    print(f"✓ Using catalog: {config.catalog}")
    
    return spark, config
