"""Snowflake configuration"""
import sys
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


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
        
        # Authentication: Support both password and key pair authentication
        # Option 1: Password authentication (sys.argv[5] = password)
        # Option 2: Key pair authentication (sys.argv[5] = "KEY_PAIR", sys.argv[6] = private_key, sys.argv[7] = passphrase (optional))
        if len(sys.argv) > 5:
            auth_method = sys.argv[5]
            if auth_method == "KEY_PAIR":
                # Key pair authentication
                self.auth_method = "key_pair"
                if len(sys.argv) > 6:
                    # Private key can be base64 encoded or raw PEM
                    private_key_str = sys.argv[6]
                    try:
                        # Try to decode if base64 encoded
                        private_key_bytes = base64.b64decode(private_key_str)
                    except Exception:
                        # If not base64, assume it's raw PEM string
                        private_key_bytes = private_key_str.encode('utf-8')
                    
                    # Parse the private key
                    try:
                        self.private_key = serialization.load_pem_private_key(
                            private_key_bytes,
                            password=None,  # Will be set if passphrase provided
                            backend=default_backend()
                        )
                    except Exception as e:
                        raise ValueError(f"Invalid private key format: {e}")
                    
                    # Optional passphrase for encrypted private key
                    if len(sys.argv) > 7:
                        passphrase = sys.argv[7]
                        try:
                            self.private_key = serialization.load_pem_private_key(
                                private_key_bytes,
                                password=passphrase.encode('utf-8'),
                                backend=default_backend()
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid passphrase or private key: {e}")
                    else:
                        self.private_key_passphrase = None
                else:
                    raise ValueError("SNOWFLAKE_PRIVATE_KEY is required for key pair authentication. Set it as job parameter (sys.argv[6]).")
            else:
                # Password authentication (default)
                self.auth_method = "password"
                self.password = auth_method  # sys.argv[5] is the password
                self.private_key = None
                self.private_key_passphrase = None
        else:
            raise ValueError("SNOWFLAKE_PASSWORD or KEY_PAIR authentication is required. Set it as job parameter (sys.argv[5]).")
        
        # Adjust parameter indices based on authentication method
        if self.auth_method == "key_pair":
            # For key pair: sys.argv[6] = private_key, sys.argv[7] = passphrase (optional)
            # So warehouse starts at sys.argv[7] or sys.argv[8]
            warehouse_idx = 7 if len(sys.argv) <= 8 else 8
            database_idx = warehouse_idx + 1
            schema_idx = database_idx + 1
        else:
            # For password: sys.argv[5] = password
            # So warehouse starts at sys.argv[6]
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
        params = {
            "account": self.account,
            "user": self.user,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema
        }
        
        if self.auth_method == "key_pair":
            params["private_key"] = self.private_key
            if self.private_key_passphrase:
                params["private_key_passphrase"] = self.private_key_passphrase
        else:
            params["password"] = self.password
        
        return params