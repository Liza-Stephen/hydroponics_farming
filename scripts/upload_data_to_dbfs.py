"""
Script to upload raw data to DBFS
Run this locally before creating the Databricks job
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from databricks_cli.dbfs.api import DbfsApi
    from databricks_cli.configure.provider import ProfileConfigProvider
    DATABRICKS_CLI_AVAILABLE = True
except ImportError:
    DATABRICKS_CLI_AVAILABLE = False
    print("ERROR: databricks-cli is required. Install via: brew install databricks/tap/databricks")
    sys.exit(1)


def upload_file_to_dbfs(local_path, dbfs_path):
    """Upload a file to DBFS"""
    try:
        # Get Databricks CLI config
        config_provider = ProfileConfigProvider()
        config = config_provider.get_config()
        
        if not config:
            print("ERROR: Databricks CLI not configured. Run: databricks configure")
            sys.exit(1)
        
        # Create DbfsApi client
        dbfs_api = DbfsApi(config)
        
        # Upload file
        print(f"Uploading {local_path} to {dbfs_path}...")
        with open(local_path, 'rb') as f:
            dbfs_api.put_file(dbfs_path, f, overwrite=True)
        
        print(f"✓ Successfully uploaded to {dbfs_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to upload file: {str(e)}")
        return False


def main():
    """Main function to upload data to DBFS"""
    # Paths
    local_data_file = project_root / "raw_data" / "iot_data_raw.csv"
    dbfs_data_path = "dbfs:/mnt/hydroponics/raw_data/iot_data_raw.csv"
    
    if not local_data_file.exists():
        print(f"ERROR: Source file not found: {local_data_file}")
        sys.exit(1)
    
    print("="*60)
    print("UPLOAD DATA TO DBFS")
    print("="*60)
    print(f"Local file: {local_data_file}")
    print(f"DBFS destination: {dbfs_data_path}")
    print()
    
    # Upload the file
    success = upload_file_to_dbfs(str(local_data_file), dbfs_data_path)
    
    if success:
        print("\n" + "="*60)
        print("✓ Data upload completed successfully!")
        print("="*60)
        print(f"\nYou can now run the pipeline in Databricks.")
        print(f"Set SOURCE_DATA_PATH={dbfs_data_path} in your Databricks job configuration.")
    else:
        print("\n" + "="*60)
        print("✗ Data upload failed!")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
