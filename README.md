# Hydroponics Farming Data Processing Pipeline

A Databricks-based data processing pipeline implementing the medallion architecture (Bronze → Silver → Gold → Snowflake) for IoT sensor data from hydroponics farming systems.

## Architecture

**Job Structure:**
- **Single Job** with four dependent tasks:
  - **Bronze Task**: Runs `bronze/batch_ingestion.py` - raw data ingestion
  - **Silver Task**: Runs `silver/processing.py` - data cleaning and validation (depends on Bronze)
  - **Gold Task**: Runs `gold/processing.py` - fact and dimension tables (depends on Silver)
  - **Snowflake Task**: Runs `snowflake/processing.py` - load data to Snowflake (depends on Gold)

**Execution:**
- Code runs natively in Databricks via Jobs
- **Data Storage**: 
  - Raw data read from S3
  - Parquet files written to S3 for each layer (bronze, silver, gold)
  - Delta tables written to Databricks Unity Catalog
- Tasks run sequentially with dependencies (Bronze → Silver → Gold)
- Job parameters: `DATABRICKS_CATALOG` and `SOURCE_DATA_PATH` (S3 path)

## Project Structure

```
hydroponics_farming/
├── config/
│   ├── databricks_config.py      # Databricks configuration
│   └── snowflake_config.py       # Snowflake configuration
├── src/
│   ├── data_processing/
│   │   ├── bronze/
│   │   │   ├── batch_ingestion.py          # Bronze layer batch ingestion
│   │   │   └── api_ingestion.py             # Bronze layer API ingestion
│   │   ├── silver/
│   │   │   └── processing.py      # Silver layer cleaning & validation
│   │   ├── gold/
│   │   │   └── processing.py      # Gold layer fact & dimension tables
│   │   ├── snowflake/
│   │   │   └── processing.py      # Snowflake layer - load data to Snowflake
│   │   └── main.py                # Main orchestration script (optional)
├── scripts/
│   ├── split_data_by_date_ranges.py # Split CSV by date ranges (backfill/incremental/API)
│   ├── batch_ingestion_example.py   # Example batch processing workflow
│   ├── api_server.py                # API server for receiving sensor data
│   └── api_simulator.py             # Simulate API-based data ingestion
├── jobs/
│   └── data_processing.json       # Main job configuration (Bronze → Silver → Gold → Snowflake)
│   └── bronze_processing.json       # Batch and API ingestion into bronze layer
├── raw_data/
│   └── iot_data_raw.csv          # Source IoT sensor data (tracked with DVC)
├── .dvc/
│   ├── config                     # DVC configuration
│   └── .gitignore                 # DVC gitignore
├── .dvcignore                     # DVC ignore patterns
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Databricks workspace access
- Databricks CLI (for CLI-based job creation)
- Python 3.8+ with pip
- DVC (Data Version Control) for data versioning

### Installation

```bash
# Install dependencies for local development (API server, DVC, etc.)
pip install -r requirements.txt
```

- Flask, requests, boto3 (for API server simulation)
- DVC with S3 support (for data versioning)


### Setup Steps

#### 1. Configure Databricks CLI

```bash
databricks configure
# Enter workspace URL and personal access token
```

#### 4. Create Databricks Job
1. Go to **Workflows** → **Jobs** → **Create Job**
2. Click **JSON Editor** (or configure manually)
3. Copy contents from `jobs/data_processing.json`
4. **Important**: Update the Repo paths
5. Click **Create**

The job automatically sets up dependencies: Bronze → Silver → Gold → Snowflake

#### 5. Configure Job Parameters (Required)

The job requires the following parameters (no defaults):

**For Bronze, Silver, and Gold tasks:**
- `DATABRICKS_CATALOG`: Unity Catalog catalog name (e.g., `hydroponics`, `main`)
- `SOURCE_FILE_KEY`: S3 file key (path within bucket) for source CSV file (e.g., `bronze/raw_data/iot_data_raw.csv`) - **Bronze task only**
- `S3_BUCKET`: S3 bucket name (e.g., `hydroponics-data`)

**For Snowflake task (additional parameters):**

**Authentication:**
- `SNOWFLAKE_ACCOUNT`: Snowflake account identifier
- `SNOWFLAKE_USER`: Snowflake username
- `SNOWFLAKE_PASSWORD`: Snowflake password
- `SNOWFLAKE_WAREHOUSE`: Snowflake warehouse name
- `SNOWFLAKE_DATABASE`: Snowflake database name
- `SNOWFLAKE_SCHEMA`: Snowflake schema name

**To set parameters when running:**

**UI**: 
1. Click **Run Now** → **Parameters** 
2. Add parameters:
   ```json
   {
     "DATABRICKS_CATALOG": "hydroponics",
     "SOURCE_FILE_KEY": "bronze/raw_data/iot_data_raw.csv",
     "S3_BUCKET": "hydroponics-data",
     "SNOWFLAKE_ACCOUNT": "account_identifier",
     "SNOWFLAKE_USER": "your_username",
     "SNOWFLAKE_PASSWORD": "your_password",
     "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
     "SNOWFLAKE_DATABASE": "HYDROPONICS_DB",
     "SNOWFLAKE_SCHEMA": "ANALYTICS"
   }
   ```

**CLI**: 
```bash
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "hydroponics",
    "SOURCE_FILE_KEY": "bronze/raw_data/iot_data_raw.csv",
    "S3_BUCKET": "hydroponics-data",
    "SNOWFLAKE_ACCOUNT": "xy12345.us-east-1",
    "SNOWFLAKE_USER": "your_username",
    "SNOWFLAKE_PASSWORD": "your_password",
    "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
    "SNOWFLAKE_DATABASE": "HYDROPONICS_DB",
    "SNOWFLAKE_SCHEMA": "ANALYTICS"
  }
}'
```

**Note**: The `SOURCE_FILE_KEY` should be the path within the S3 bucket (without the `s3://` prefix or bucket name). Leading slashes are automatically removed.

#### 5. Setup Snowflake (Required for Snowflake Layer)

1. **Create a Snowflake User** (in Snowflake UI or SQL):
   ```sql
   -- Connect to Snowflake as ACCOUNTADMIN
   CREATE USER databricks_user 
     PASSWORD = 'your_secure_password'
     DEFAULT_ROLE = 'PUBLIC'
     DEFAULT_WAREHOUSE = 'HYDRO_WH';
   
   -- Grant necessary permissions
   GRANT USAGE ON WAREHOUSE HYDRO_WH TO ROLE PUBLIC;
   GRANT CREATE DATABASE ON ACCOUNT TO ROLE PUBLIC;
   ```

2. **Create a Warehouse** (if not exists):
   ```sql
   CREATE WAREHOUSE IF NOT EXISTS HYDRO_WH
     WITH WAREHOUSE_SIZE = 'X-SMALL'
     AUTO_SUSPEND = 60
     AUTO_RESUME = TRUE;
   ```

3. **Note Your Credentials:**
   - Account identifier: from your Snowflake account
   - Username: `databricks_user` (or whatever you created)
   - Password: The password you set

4. **Use These Credentials in Databricks Job Parameters:**
   - Set `SNOWFLAKE_ACCOUNT` = your account identifier
   - Set `SNOWFLAKE_USER` = your Snowflake username
   - Set `SNOWFLAKE_PASSWORD` = your Snowflake password

**Note**: 
- The Snowflake layer will automatically create the database and schema if they don't exist
- Store sensitive credentials (password) in Databricks Secrets for security

#### 6. Run the Job

**From Databricks UI:**
1. Go to **Workflows** → **Jobs**
2. Find your `data_processing` job
3. Click **Run Now**

The job will execute tasks in sequence: Bronze → Silver → Gold → Snowflake

#### 7. Verify Results

**Databricks:**
```sql
USE CATALOG hydroponics;
SELECT COUNT(*) FROM bronze.iot_data;
SELECT COUNT(*) FROM silver.iot_data;
SELECT COUNT(*) FROM gold.iot_data;
SELECT COUNT(*) FROM gold.dim_time;
SELECT COUNT(*) FROM gold.dim_equipment;
```

**Snowflake:**
```sql
USE DATABASE HYDROPONICS_DB;
USE SCHEMA ANALYTICS;
SELECT COUNT(*) FROM dim_time;
SELECT COUNT(*) FROM dim_equipment;
SELECT COUNT(*) FROM iot_data;
```

## Data Version Control (DVC)

This project uses DVC (Data Version Control) for versioning large data files that shouldn't be stored directly in Git.

### DVC Setup

1. **Install DVC**:
   ```bash
   pip install "dvc[s3]"
   ```

2. **Initialize DVC**:
   ```bash
   dvc init
   ```

3. **Configure Remote Storage**:
   ```bash
   # S3 remote
   dvc remote add -d s3 s3://your-bucket/dvc-storage
   ```

### Tracking Data Files

**Track local data files with DVC:**

```bash
# Track raw data file (local)
dvc add raw_data/iot_data_raw.csv

# Commit DVC metadata files to Git
git add raw_data/iot_data_raw.csv.dvc data_splits.dvc api_data.dvc .dvc .gitignore
git commit -m "Add data files with DVC"
```

**Reproduce pipeline with specific data version:**
```bash
# Checkout specific data version
git checkout <commit-hash>
dvc pull  # Pull corresponding data files from S3
```

## Configuration

### Job Parameters

The job requires the following parameters (no hardcoded defaults):
- **DATABRICKS_CATALOG**: Unity Catalog catalog name (required)
- **SOURCE_DATA_PATH**: S3 path to source CSV file (required for bronze task)
- **S3_BUCKET**: S3 bucket name (required, or will be extracted from SOURCE_DATA_PATH if it's an S3 path)

Parameters are passed to Python scripts via `sys.argv`:
- Bronze task: `[DATABRICKS_CATALOG, S3_BUCKET, SOURCE_FILE_KEY]`
- Silver task: `[DATABRICKS_CATALOG, S3_BUCKET]`
- Gold task: `[DATABRICKS_CATALOG, S3_BUCKET]`
- Snowflake task: `[DATABRICKS_CATALOG, S3_BUCKET, SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA]`

**Note**: If `S3_BUCKET` is not provided as a parameter, the system will attempt to extract it from `SOURCE_DATA_PATH` if it's an S3 path (e.g., `s3://bucket-name/path` → `bucket-name`).

### Data Storage

**S3 Storage (Parquet Files):**
- `s3://{bucket}/bronze/parquet/iot_data/` - Raw ingested data (parquet)
- `s3://{bucket}/silver/parquet/iot_data/` - Cleaned and validated data (parquet)
- `s3://{bucket}/gold/parquet/iot_data/` - Fact table (parquet)
- `s3://{bucket}/gold/parquet/dim_time/` - Time dimension (parquet)
- `s3://{bucket}/gold/parquet/dim_equipment/` - Equipment dimension (parquet)

**Databricks Tables (Delta):**
- `{catalog}.bronze.iot_data` - Raw ingested data
- `{catalog}.silver.iot_data` - Cleaned and validated data
- `{catalog}.gold.iot_data` - Fact table
- `{catalog}.gold.dim_time` - Time dimension
- `{catalog}.gold.dim_equipment` - Equipment dimension

## Data Processing Details

### Bronze Layer
- Ingests raw CSV from S3
- Writes parquet files to S3 (`s3://{bucket}/bronze/parquet/iot_data/`)
- Writes Delta table to Databricks Unity Catalog
- Preserves all original data
- Adds metadata: `ingestion_timestamp`, `source_file`
- Uses append mode for incremental loads

### Silver Layer
- Reads parquet files from S3 (bronze layer) or falls back to Databricks table
- Writes parquet files to S3 (`s3://{bucket}/silver/parquet/iot_data/`)
- Writes Delta table to Databricks Unity Catalog
- Type conversions (string to numeric, ON/OFF to boolean)
- Data quality validation (pH, TDS, temperature, humidity ranges)
- Deduplication based on id and timestamp
- Adds `silver_processed_timestamp`

### Gold Layer
- Reads parquet files from S3 (silver layer) or falls back to Databricks table
- Writes parquet files to S3 for all tables (`s3://{bucket}/gold/parquet/`)
- Writes Delta tables to Databricks Unity Catalog
- Creates star schema with fact and dimension tables:
  - `dim_time` - Time dimension for temporal analysis
  - `dim_equipment` - Equipment dimension for equipment tracking
  - `iot_data` - Fact table with sensor readings and optimal condition indicators
- Calculates optimal condition indicators (pH, TDS, temperature, humidity)

### Snowflake Layer
- **Source**: Reads from **Gold layer output** (fact and dimension tables)
- Reads parquet files from S3 (`s3://{bucket}/gold/parquet/`) or falls back to Gold Databricks tables
- Loads the following Gold tables to Snowflake:
  - `dim_time` - Time dimension
  - `dim_equipment` - Equipment dimension
  - `iot_data` - Fact table with sensor readings
- Writes to Snowflake database for analytics and reporting
- Uses batch inserts for data loading

## Data Ingestion Patterns

### Date-Based Data Splitting

Split data by date ranges for different ingestion patterns:

```bash
# Split data into Batch Backfill, Incremental Batch, and API/Streaming Replay
python scripts/split_data_by_date_ranges.py \
    --input raw_data/iot_data_raw.csv \
    --output-dir data_splits/
```

This creates:
- **Batch Backfill** (2023-11-26 to 2023-12-19): Single file with ~24 days of historical data
  - Use for initial data load: `batch_backfill_2023-11-26_to_2023-12-19.csv`
- **Incremental Batch** (2023-12-20 to 2023-12-23): Daily files for incremental processing
  - Process one file per day: `incremental_YYYY-MM-DD.csv`
  - Simulates daily batch ingestion workflow
- **API/Streaming Replay** (2023-12-24 to 2023-12-26): Daily files for event replay
  - Use for API simulation or streaming replay: `api_replay_YYYY-MM-DD.csv`
  - Can be sent via API simulator or processed as streaming events

**Output structure:**
```
data_splits/
├── batch_backfill/
│   └── batch_backfill_2023-11-26_to_2023-12-19.csv
├── incremental_batch/
│   ├── incremental_2023-12-20.csv
│   ├── incremental_2023-12-21.csv
│   └── ... (daily files)
└── api_streaming_replay/
    ├── api_replay_2023-12-24.csv
    ├── api_replay_2023-12-25.csv
    └── api_replay_2023-12-26.csv
```

### Ingestion Workflow Example

**1. Batch Backfill (Historical Data):**
```bash
# Run bronze ingestion once for the entire backfill period
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "your_catalog",
    "SOURCE_FILE_KEY": "bronze/batch_backfill/batch_backfill_2023-11-26_to_2023-12-19.csv",
    "S3_BUCKET": "your-bucket"
  }
}'
```

**2. Incremental Batch (Daily Processing):**
```bash
# Process each daily file sequentially
for date in 2023-12-20 2023-12-21 2023-12-22 2023-12-23; do
  databricks jobs run-now <job-id> --json "{
    \"job_parameters\": {
      \"DATABRICKS_CATALOG\": \"your_catalog\",
      \"SOURCE_FILE_KEY\": \"bronze/incremental_batch/incremental_${date}.csv\",
      \"S3_BUCKET\": \"your-bucket\"
    }
  }"
```

**3. API/Streaming Replay:**
```bash
# Option A: Send via API simulator
python scripts/api_simulator.py \
    --input data_splits/api_streaming_replay/api_replay_2023-12-24.csv \
    --api-url http://localhost:8000/api/sensor-data \
    --delay 0.1

# Option B: Process as batch (if replaying from files)
# First upload JSON files to S3
aws s3 cp --recursive api_data/ s3://your-bucket/bronze/api_data/

# Then run API ingestion job
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "your_catalog",
    "SOURCE_FILE_KEY": "bronze/api_data/",
    "S3_BUCKET": "your-bucket"
  }
}'
```

### API-Based Ingestion

Simulate real-time IoT sensor data ingestion via API:

#### Prerequisites

Before starting, install required Python packages:

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 1: Start API Server**
```bash
# Start the API server with S3 storage
python scripts/api_server.py \
    --port 8000 \
    --s3-bucket your-bucket-name \
    --s3-prefix raw/api_data/ \
    --buffer-size 100
```

You should see output like:
```
Starting API server on port 8000
S3 bucket: s3://your-bucket-name/bronze/api_data/
✓ S3 connection successful
Buffer size: 100
API endpoint: http://localhost:8000/api/sensor-data
------------------------------------------------------------
 * Running on http://0.0.0.0:8000
```

**Note:** Keep this terminal open and running. The server will continue to accept requests.

**Step 2: Simulate API Calls**

Open another terminal while keeping the previous running:

```bash
for date in 2023-12-24 2023-12-25 2023-12-26; do
  python scripts/api_simulator.py \
      --input data_splits/api_streaming_replay/api_replay_${date}.csv \
      --api-url http://localhost:8000/api/sensor-data \
      --delay 0.1
done
```

**Step 3: Check Received Data**

**If using local file storage:**
```bash
# View JSON files created by API server
ls -lh api_data/

# View a sample file
cat api_data/sensor_data_*.json | head -20
```

**If using S3 storage:**
```bash
# List files in S3
aws s3 ls s3://your-bucket-name/bronze/api_data/ --recursive

# View a sample file
aws s3 cp s3://your-bucket-name/bronze/api_data/sensor_data_*.json - | head -20
```

**Step 4: Stop API Server**

Terminate execution in first terminal (Ctrl+C)

**Step 5: Run API Ingestion Job**

**If using local file storage:**
```bash
# First upload JSON files to S3
aws s3 cp --recursive api_data/ s3://your-bucket/bronze/api_data/
```

**Then run the API ingestion job:**
```bash
databricks jobs run-now <api-job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "your_catalog",
    "SOURCE_FILE_KEY": "bronze/api_data/",
    "S3_BUCKET": "your-bucket"
  }
}'
```

#### API Configuration Options

**API Server Options:**
- `--port`, `-p`: Server port (default: `8000`)
- `--output-dir`, `-o`: Output directory for local file storage
- `--s3-bucket`: S3 bucket name for direct S3 storage (takes precedence over `--output-dir`)
- `--s3-prefix`: S3 prefix/path for storing data (default: `bronze/api_data/`)
- `--buffer-size`, `-b`: Buffer size before flushing (default: `100`)

**API Simulator Options:**
- `--input`, `-i`: CSV file to read from (required)
- `--api-url`, `-u`: API endpoint URL (required)
- `--delay`, `-d`: Delay between requests in seconds (default: `0.1`)
- `--batch-size`, `-b`: Number of records per request (default: `1`)
- `--start-from`: Row number to start from (0-indexed)
- `--max-records`, `-m`: Maximum number of records to send

**API Server Options:**
- `--port`: Server port (default: `8000`)
- `--output-dir`: Directory for JSON files (default: `api_data/`)
- `--buffer-size`: Buffer size before flushing (default: `100`)

#### API Endpoints

- `POST /api/sensor-data` - Send single record or batch of records
- `POST /api/flush` - Manually flush buffered records
- `GET /health` - Health check

#### API Data Format

```json
// Single record
{
  "id": "1",
  "timestamp": "2023-11-26 10:57:52",
  "pH": "7",
  "TDS": "500",
  "water_level": "0",
  "DHT_temp": "25.5",
  "DHT_humidity": "60",
  "water_temp": "20",
  "pH_reducer": "ON",
  "add_water": null,
  "nutrients_adder": "OFF",
  "humidifier": "OFF",
  "ex_fan": "ON"
}

// Batch of records
{
  "records": [
    {"id": "1", "timestamp": "...", ...},
    {"id": "2", "timestamp": "...", ...}
  ]
}
```

## Querying the Data

```sql
-- Example: Find readings with suboptimal pH
SELECT 
    t.date,
    t.hour,
    f.ph_level,
    f.is_ph_optimal,
    f.air_temperature
FROM hydroponics.gold.iot_data f
JOIN hydroponics.gold.dim_time t
    ON f.timestamp_key = t.timestamp_key
WHERE f.is_ph_optimal = false
ORDER BY t.date DESC, t.hour DESC;
```
