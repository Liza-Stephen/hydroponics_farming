"""
dbt Runner for Databricks
Runs dbt models as part of the Databricks job pipeline
"""
import sys
import os
import subprocess
from pathlib import Path

def run_dbt_models(
    snowflake_account,
    snowflake_user,
    snowflake_password,
    snowflake_warehouse,
    snowflake_database,
    snowflake_schema,
    dbt_target="dev"
):
    """
    Run dbt models from Databricks
    
    Args:
        snowflake_account: Snowflake account identifier
        snowflake_user: Snowflake username
        snowflake_password: Snowflake password
        snowflake_warehouse: Snowflake warehouse name
        snowflake_database: Snowflake database name
        snowflake_schema: Snowflake schema name (source schema)
        dbt_target: dbt target environment (dev or prod)
    """
    # Get project root (assuming this is in src/data_processing/semantic_layer)
    project_root = Path(__file__).resolve().parents[3]
    dbt_project_path = project_root / "dbt"
    
    if not dbt_project_path.exists():
        raise ValueError(f"dbt project not found at {dbt_project_path}")
    
    # Set environment variables for dbt
    os.environ["SNOWFLAKE_ACCOUNT"] = snowflake_account
    os.environ["SNOWFLAKE_USER"] = snowflake_user
    os.environ["SNOWFLAKE_PASSWORD"] = snowflake_password
    os.environ["SNOWFLAKE_WAREHOUSE"] = snowflake_warehouse
    os.environ["SNOWFLAKE_DATABASE"] = snowflake_database
    os.environ["SNOWFLAKE_SCHEMA"] = snowflake_schema
    
    print(f"Running dbt models from: {dbt_project_path}")
    print(f"Target: {dbt_target}")
    print(f"Source schema: {snowflake_schema}")
    
    # Change to dbt project directory
    os.chdir(dbt_project_path)
    
    try:
        # Run dbt commands
        commands = [
            ["dbt", "deps"],  # Install dependencies
            ["dbt", "run", "--target", dbt_target],  # Run all models
            ["dbt", "test", "--target", dbt_target],  # Run tests
        ]
        
        for cmd in commands:
            print(f"\nExecuting: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            print(result.stdout)
            if result.stderr:
                print("Errors/Warnings:", result.stderr)
            
            if result.returncode != 0:
                raise RuntimeError(f"dbt command failed: {' '.join(cmd)}")
        
        print("\n✓ dbt models completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running dbt: {e}")
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "dbt not found. Install dbt-snowflake in the Databricks environment:\n"
            "Add 'dbt-snowflake>=1.6.0' to your Databricks environment dependencies"
        )


if __name__ == "__main__":
    # Read parameters from sys.argv (Databricks job parameters)
    # Parameters: [SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    #              SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, DBT_TARGET]
    
    if len(sys.argv) < 7:
        raise ValueError(
            "Missing required parameters. Expected:\n"
            "  sys.argv[1] = SNOWFLAKE_ACCOUNT\n"
            "  sys.argv[2] = SNOWFLAKE_USER\n"
            "  sys.argv[3] = SNOWFLAKE_PASSWORD\n"
            "  sys.argv[4] = SNOWFLAKE_WAREHOUSE\n"
            "  sys.argv[5] = SNOWFLAKE_DATABASE\n"
            "  sys.argv[6] = SNOWFLAKE_SCHEMA\n"
            "  sys.argv[7] = DBT_TARGET (optional, default: dev)"
        )
    
    snowflake_account = sys.argv[1]
    snowflake_user = sys.argv[2]
    snowflake_password = sys.argv[3]
    snowflake_warehouse = sys.argv[4]
    snowflake_database = sys.argv[5]
    snowflake_schema = sys.argv[6]
    dbt_target = sys.argv[7] if len(sys.argv) > 7 else "dev"
    
    run_dbt_models(
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_password=snowflake_password,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema,
        dbt_target=dbt_target
    )
