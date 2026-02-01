"""Snowflake configuration"""
import sys


class SnowflakeConfig:
    """Configuration for Snowflake connection"""
    
    def __init__(self):
        # Read from job parameters (sys.argv) only - no environment variables
        # Job parameters for Snowflake task:
        #   sys.argv[1] = DATABRICKS_CATALOG (for reading from Gold layer)
        #   sys.argv[2] = S3_BUCKET (for reading Gold parquet files)
        #   sys.argv[3] = SNOWFLAKE_ACCOUNT
        #   sys.argv[4] = SNOWFLAKE_USER
        #   sys.argv[5] = SNOWFLAKE_PASSWORD
        #   sys.argv[6] = SNOWFLAKE_WAREHOUSE
        #   sys.argv[7] = SNOWFLAKE_DATABASE
        #   sys.argv[8] = SNOWFLAKE_SCHEMA
        
        # Databricks catalog (for reading Gold layer)
        if len(sys.argv) > 1:
            self.databricks_catalog = sys.argv[1]
        else:
            raise ValueError("DATABRICKS_CATALOG is required. Set it as job parameter (sys.argv[1]).")
        
        # S3 bucket (for reading Gold parquet files)
        if len(sys.argv) > 2:
            self.s3_bucket = sys.argv[2]
            self.s3_gold_path = f"s3://{self.s3_bucket}/gold"
        else:
            raise ValueError("S3_BUCKET is required. Set it as job parameter (sys.argv[2]).")
        
        # Snowflake connection parameters
        if len(sys.argv) > 3:
            self.account = sys.argv[3]
        else:
            raise ValueError("SNOWFLAKE_ACCOUNT is required. Set it as job parameter (sys.argv[3]).")
        
        if len(sys.argv) > 4:
            self.user = sys.argv[4]
        else:
            raise ValueError("SNOWFLAKE_USER is required. Set it as job parameter (sys.argv[4]).")
        
        # Password authentication (sys.argv[5] = password)
        if len(sys.argv) > 5:
            self.password = sys.argv[5]
        else:
            raise ValueError("SNOWFLAKE_PASSWORD is required. Set it as job parameter (sys.argv[5]).")
        
        # Parameter indices for password authentication:
        # sys.argv[5] = password
        # sys.argv[6] = warehouse
        # sys.argv[7] = database
        # sys.argv[8] = schema
        warehouse_idx = 6
        database_idx = 7
        schema_idx = 8
        
        if len(sys.argv) > warehouse_idx:
            self.warehouse = sys.argv[warehouse_idx]
        else:
            raise ValueError("SNOWFLAKE_WAREHOUSE is required. Set it as job parameter.")
        
        if len(sys.argv) > database_idx:
            self.database = sys.argv[database_idx]
        else:
            raise ValueError("SNOWFLAKE_DATABASE is required. Set it as job parameter.")
        
        if len(sys.argv) > schema_idx:
            self.schema = sys.argv[schema_idx]
        else:
            raise ValueError("SNOWFLAKE_SCHEMA is required. Set it as job parameter.")
        
        # Gold schema name in Databricks
        self.gold_schema = "gold"
    
    def get_gold_table_name(self, table):
        """Get full Gold table name: catalog.schema.table"""
        return f"{self.databricks_catalog}.{self.gold_schema}.{table}"
    
    def get_snowflake_table_name(self, table):
        """Get full Snowflake table name: database.schema.table"""
        return f"{self.database}.{self.schema}.{table}"
    
    def get_connection_params(self):
        """Get Snowflake connection parameters"""
        return {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema
        }