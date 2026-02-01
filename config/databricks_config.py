"""Databricks configuration"""
import sys
from pyspark.sql import SparkSession


class DatabricksConfig:
    """Configuration for Databricks"""
    
    def __init__(self):
        # Read from job parameters (sys.argv) only - no environment variables
        # Job parameters: 
        #   sys.argv[1] = DATABRICKS_CATALOG
        #   sys.argv[2] = S3_BUCKET (all tasks)
        #   sys.argv[3] = SOURCE_FILE_KEY (bronze only)
        
        # Catalog (required)
        if len(sys.argv) > 1:
            self.catalog = sys.argv[1]
        else:
            raise ValueError("DATABRICKS_CATALOG is required. Set it as job parameter (sys.argv[1]).")
        
        # S3 bucket (required) - second parameter for all tasks
        if len(sys.argv) > 2:
            self.s3_bucket = sys.argv[2]
        else:
            raise ValueError("S3_BUCKET is required. Set it as job parameter (sys.argv[2]).")
        
        # Read SOURCE_FILE_KEY from parameters (required for bronze, not used for silver/gold)
        # Construct full S3 path: s3://bucket/file_key
        if len(sys.argv) > 3:
            # Bronze task: sys.argv[3] = SOURCE_FILE_KEY
            file_key = sys.argv[3]
            # Remove leading slash if present
            if file_key.startswith("/"):
                file_key = file_key[1:]
            self.source_data_path = f"s3://{self.s3_bucket}/{file_key}"
        else:
            self.source_data_path = None
        
        self.s3_bronze_path = f"s3://{self.s3_bucket}/bronze"
        self.s3_silver_path = f"s3://{self.s3_bucket}/silver"
        self.s3_gold_path = f"s3://{self.s3_bucket}/gold"
        
        self.bronze_schema = "bronze"
        self.silver_schema = "silver"
        self.gold_schema = "gold"
    
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
