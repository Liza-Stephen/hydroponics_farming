# Hydroponics Farming Data Processing Pipeline

A Databricks-based data processing pipeline implementing the medallion architecture (Bronze → Silver → Gold) for IoT sensor data from hydroponics farming systems.

## Architecture

**Job Structure:**
- **Single Job** with three dependent tasks:
  - **Bronze Task**: Runs `bronze_ingestion.py` - raw data ingestion
  - **Silver Task**: Runs `silver_processing.py` - data cleaning and validation (depends on Bronze)
  - **Gold Task**: Runs `gold_processing.py` - fact and dimension tables (depends on Silver)

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
│   │   ├── bronze_ingestion.py   # Bronze layer ingestion
│   │   ├── silver_processing.py  # Silver layer cleaning & validation
│   │   └── gold_processing.py    # Gold layer fact & dimension tables
│   └── main.py                   # Main orchestration script (optional)
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
