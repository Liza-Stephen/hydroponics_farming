# Hydroponics Farming Data Processing Pipeline

A Databricks-based data processing pipeline implementing the medallion architecture (Bronze → Silver → Gold) for IoT sensor data from hydroponics farming systems.

## Architecture

**Job Structure:**
- **Single Job** with three dependent tasks:
  - **Bronze Task**: Runs `bronze/batch_ingestion.py` - raw data ingestion
  - **Silver Task**: Runs `silver/processing.py` - data cleaning and validation (depends on Bronze)
  - **Gold Task**: Runs `gold/processing.py` - fact and dimension tables (depends on Silver)

**Execution:**
- Code runs natively in Databricks via Jobs
- Data stored in Unity Catalog Volumes
- Tasks run sequentially with dependencies (Bronze → Silver → Gold)
- Job parameters: `DATABRICKS_CATALOG` and `SOURCE_DATA_PATH`

## Project Structure

```
hydroponics_farming/
├── config/
│   ├── databricks_config.py      # Databricks configuration
├── src/
│   ├── data_processing/
│   │   ├── bronze/
│   │   │   ├── batch_ingestion.py          # Bronze layer batch ingestion
│   │   │   └── api_ingestion.py             # Bronze layer API ingestion
│   │   ├── silver/
│   │   │   └── processing.py      # Silver layer cleaning & validation
│   │   └── gold/
│   │   │   └── processing.py      # Gold layer fact & dimension tables
│   │   └── main.py                # Main orchestration script (optional)
├── scripts/
│   ├── split_data_by_date_ranges.py # Split CSV by date ranges (backfill/incremental/API)
│   ├── batch_ingestion_example.py   # Example batch processing workflow
│   ├── api_server.py                # API server for receiving sensor data
│   └── api_simulator.py             # Simulate API-based data ingestion
├── jobs/
│   └── data_processing.json       # Main job configuration (Bronze → Silver → Gold)
├── raw_data/
│   └── iot_data_raw.csv          # Source IoT sensor data
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Databricks workspace access
- Databricks CLI (for CLI-based job creation)

### Setup Steps

#### 1. Configure Databricks CLI

```bash
databricks configure
# Enter workspace URL and personal access token
```

#### 2. Create Databricks Job

**Option A: Using Databricks CLI**
```bash
databricks jobs create --json @jobs/data_processing.json
```

**Option B: Using Databricks UI (Recommended)**
1. Go to **Workflows** → **Jobs** → **Create Job**
2. Click **JSON Editor** (or configure manually)
3. Copy contents from `jobs/data_processing.json`
4. **Important**: Update the Repo paths
5. Click **Create**

The job automatically sets up dependencies: Bronze → Silver → Gold

#### 3. Configure Job Parameters (if needed)

The job has default parameters defined in the JSON:
- `DATABRICKS_CATALOG`: `hydroponics`
- `SOURCE_DATA_PATH`: `/Volumes/hydroponics/bronze/raw_data/iot_data_raw.csv`

To override when running:
- **UI**: Click **Run Now** → **Parameters** → Edit values
- **CLI**: `databricks jobs run-now <job-id> --json '{"job_parameters": {"DATABRICKS_CATALOG": "your_catalog", "SOURCE_DATA_PATH": "your_path"}}'`

#### 4. Run the Job

**From Databricks UI:**
1. Go to **Workflows** → **Jobs**
2. Find your `data_processing` job
3. Click **Run Now**

The job will execute tasks in sequence: Bronze → Silver → Gold

#### 5. Verify Results

```sql
USE CATALOG hydroponics;
SELECT COUNT(*) FROM bronze.iot_data;
SELECT COUNT(*) FROM silver.iot_data;
SELECT COUNT(*) FROM gold.iot_data;
SELECT COUNT(*) FROM gold.dim_time;
SELECT COUNT(*) FROM gold.dim_equipment;
```

## Configuration

### Job Parameters

The job accepts parameters defined in `jobs/data_processing.json`:
- **DATABRICKS_CATALOG**: Unity Catalog catalog name
- **SOURCE_DATA_PATH**: Path to source CSV in Volumes

Parameters are passed to Python scripts via `sys.argv`:
- Bronze task: `[DATABRICKS_CATALOG, SOURCE_DATA_PATH]`
- Silver task: `[DATABRICKS_CATALOG]`
- Gold task: `[DATABRICKS_CATALOG]`

### Data Storage

All data is stored in Unity Catalog managed tables:
- `{catalog}.bronze.iot_data` - Raw ingested data
- `{catalog}.silver.iot_data` - Cleaned and validated data
- `{catalog}.gold.iot_data` - Fact table
- `{catalog}.gold.dim_time` - Time dimension
- `{catalog}.gold.dim_equipment` - Equipment dimension

## Data Processing Details

### Bronze Layer
- Ingests raw CSV from Unity Catalog Volumes
- Preserves all original data
- Adds metadata: `ingestion_timestamp`, `source_file`
- Uses append mode for incremental loads

### Silver Layer
- Type conversions (string to numeric, ON/OFF to boolean)
- Data quality validation (pH, TDS, temperature, humidity ranges)
- Deduplication based on id and timestamp
- Adds `silver_processed_timestamp`

### Gold Layer
- Creates star schema with fact and dimension tables
- Calculates optimal condition indicators
- Time dimension for temporal analysis
- Equipment dimension for equipment tracking

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
# Upload backfill file to Databricks
# Run bronze ingestion once for the entire backfill period
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "hydroponics",
    "SOURCE_DATA_PATH": "/Volumes/hydroponics/bronze/batch_backfill/batch_backfill_2023-11-26_to_2023-12-19.csv"
  }
}'
```

**2. Incremental Batch (Daily Processing):**
```bash
# Process each daily file sequentially
for date in 2023-12-20 2023-12-21 2023-12-22 2023-12-23; do
  databricks jobs run-now <job-id> --json "{
    \"job_parameters\": {
      \"DATABRICKS_CATALOG\": \"hydroponics\",
      \"SOURCE_DATA_PATH\": \"/Volumes/hydroponics/bronze/incremental_batch/incremental_${date}.csv\"
    }
  }"
done
```

**3. API/Streaming Replay:**
```bash
# Option A: Send via API simulator
python scripts/api_simulator.py \
    --input data_splits/api_streaming_replay/api_replay_2023-12-24.csv \
    --api-url http://localhost:8000/api/sensor-data \
    --delay 0.1

# Option B: Process as batch (if replaying from files)
databricks jobs run-now <job-id> --json '{
  "job_parameters": {
    "DATABRICKS_CATALOG": "hydroponics",
    "SOURCE_DATA_PATH": "/Volumes/hydroponics/bronze/api_streaming_replay/api_replay_2023-12-24.csv"
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

Required packages:
- `flask` - Web framework for API server
- `flask-cors` - CORS support for API
- `requests` - HTTP library for API simulator

#### Manual Setup

**Step 1: Start API Server**

Open a terminal:
```bash
# Start the API server
python scripts/api_server.py \
    --port 8000 \
    --output-dir api_data/ \
    --buffer-size 100
```

You should see output like:
```
Starting API server on port 8000
Output directory: api_data
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

```bash
# View JSON files created by API server
ls -lh api_data/

# View a sample file
cat api_data/sensor_data_*.json | head -20
```

**Step 4: Stop API Server**

Terminate execution in first terminal 1

**Step 5: Ingest API Data to Bronze (Databricks)**

```python
# In Databricks, run:
from src.data_processing.api_ingestion import run_api_ingestion

# Ingest JSON files from API server
run_api_ingestion(json_dir="/Volumes/hydroponics/bronze/api_data/")
```

#### API Configuration Options

**API Simulator Options:**
- `--input`: CSV file to read from
- `--api-url`: API endpoint URL (default: `http://localhost:8000/api/sensor-data`)
- `--delay`: Delay between requests in seconds (default: `0.1`)
- `--batch-size`: Number of records per request (default: `1`)
- `--start-from`: Row number to start from (0-indexed)
- `--max-records`: Maximum number of records to send

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
