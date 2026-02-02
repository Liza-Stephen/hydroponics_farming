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
    snowflake_schema
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
    """
    # Find dbt project - simple approach: search up from current directory
    cwd = Path(os.getcwd())
    dbt_project_path = None
    
    # Try current directory and parent directories (up to 5 levels)
    for path in [cwd] + list(cwd.parents)[:5]:
        dbt_path = path / "dbt"
        if dbt_path.exists() and (dbt_path / "dbt_project.yml").exists():
            dbt_project_path = dbt_path
            break
    
    if dbt_project_path is None:
        raise ValueError(
            f"dbt project not found. Searched from: {cwd}\n"
            f"Please ensure the dbt folder exists in the repository root."
        )
    
    print(f"Using dbt project path: {dbt_project_path}")
    
    # Set environment variables for dbt
    os.environ["SNOWFLAKE_ACCOUNT"] = snowflake_account
    os.environ["SNOWFLAKE_USER"] = snowflake_user
    os.environ["SNOWFLAKE_PASSWORD"] = snowflake_password
    os.environ["SNOWFLAKE_WAREHOUSE"] = snowflake_warehouse
    os.environ["SNOWFLAKE_DATABASE"] = snowflake_database
    os.environ["SNOWFLAKE_SCHEMA"] = snowflake_schema
    
    print(f"Running dbt models from: {dbt_project_path}")
    print(f"Source schema: {snowflake_schema}")
    
    # Change to dbt project directory
    os.chdir(dbt_project_path)
    
    # Create profiles.yml dynamically (dbt requires it for connection)
    profiles_content = f"""hydroponics_semantic:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: "{snowflake_account}"
      user: "{snowflake_user}"
      password: "{snowflake_password}"
      role: PUBLIC
      database: "{snowflake_database}"
      warehouse: "{snowflake_warehouse}"
      schema: "{snowflake_schema}"
      threads: 4
      client_session_keep_alive: false
"""
    
    profiles_file = dbt_project_path / "profiles.yml"
    with open(profiles_file, "w") as f:
        f.write(profiles_content)
    
    # Set DBT_PROFILES_DIR to point to project directory
    os.environ["DBT_PROFILES_DIR"] = str(dbt_project_path)
    
    try:
        # Run dbt commands
        commands = [
            ["dbt", "deps"],  # Install dependencies
            ["dbt", "run"],  # Run all models
            ["dbt", "test"],  # Run tests
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
        
        print("\ndbt models completed successfully")
        
        # Clean up dynamically created profiles.yml
        if profiles_file.exists():
            profiles_file.unlink()
            print("Cleaned up temporary profiles.yml")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running dbt: {e}")
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "dbt not found. Install dbt-snowflake in the Databricks environment:\n"
            "Add 'dbt-snowflake>=1.6.0' to your Databricks environment dependencies"
        )
    finally:
        # Clean up profiles.yml if it still exists
        profiles_file = dbt_project_path / "profiles.yml"
        if profiles_file.exists():
            try:
                profiles_file.unlink()
            except:
                pass


if __name__ == "__main__":
    # Read parameters from sys.argv (Databricks job parameters)
    # Parameters: [SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, 
    #              SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA]
    
    if len(sys.argv) < 7:
        raise ValueError(
            "Missing required parameters. Expected:\n"
            "  sys.argv[1] = SNOWFLAKE_ACCOUNT\n"
            "  sys.argv[2] = SNOWFLAKE_USER\n"
            "  sys.argv[3] = SNOWFLAKE_PASSWORD\n"
            "  sys.argv[4] = SNOWFLAKE_WAREHOUSE\n"
            "  sys.argv[5] = SNOWFLAKE_DATABASE\n"
            "  sys.argv[6] = SNOWFLAKE_SCHEMA"
        )
    
    snowflake_account = sys.argv[1]
    snowflake_user = sys.argv[2]
    snowflake_password = sys.argv[3]
    snowflake_warehouse = sys.argv[4]
    snowflake_database = sys.argv[5]
    snowflake_schema = sys.argv[6]
    
    run_dbt_models(
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_password=snowflake_password,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema
    )
