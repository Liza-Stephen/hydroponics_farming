# Hydroponics Farming Data Processing Pipeline

A Databricks-based data processing pipeline implementing the medallion architecture (Bronze → Silver → Gold) for IoT sensor data from hydroponics farming systems.

## Execution Mode

**Code runs natively in Databricks via Jobs.**

- Data is stored in Unity Catalog Volumes
- Pipeline runs as a Databricks Job (triggered from local via CLI)
- All execution happens natively in Databricks (no Databricks Connect needed)
- Uses serverless compute automatically (workspace requirement)

## Architecture Overview

This project implements a three-layer medallion architecture:

- **Bronze Layer**: Raw data ingestion from CSV files in Unity Catalog Volumes
- **Silver Layer**: Cleaned, validated, and deduplicated data
- **Gold Layer**: Business-ready fact and dimension tables in star schema format

## Project Structure

```
hydroponics_farming/
├── config/
│   ├── databricks_config.py      # Databricks configuration
│   └── config_template.env        # Environment variables template
├── src/
│   ├── data_processing/
│   │   ├── bronze_ingestion.py   # Bronze layer ingestion
│   │   ├── silver_processing.py  # Silver layer cleaning & validation
│   │   └── gold_processing.py    # Gold layer fact & dimension tables
│   └── main.py                   # Main orchestration script
├── scripts/
│   └── create_databricks_job.sh   # Script to create Databricks job
├── raw_data/
│   └── iot_data_raw.csv          # Source IoT sensor data
├── requirements.txt              # Python dependencies
├── databricks_job_config.json    # Databricks job configuration
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Python 3.8+ (for local scripts)
- Databricks workspace access
- Databricks CLI (`brew install databricks/tap/databricks`)

### Setup Steps

#### 1. Configure Databricks CLI

```bash
brew install databricks/tap/databricks
databricks configure
# Enter workspace URL and personal access token
```

#### 2. Create Databricks Job

**Option A: Using script**
```bash
# Edit databricks_job_config.json:
# - Update path to your Python file location (line 8)
# Note: Serverless compute is used automatically (no compute configuration needed)
bash scripts/create_databricks_job.sh
```

After creating the job, configure parameters in the Databricks UI:
1. Go to **Workflows** → **Jobs** → Your job
2. Click **Edit**
3. Go to **Task** → **Parameters**
4. Set parameters:
   - Parameter 1: `hydroponics` (DATABRICKS_CATALOG)
   - Parameter 2: `/Volumes/hydroponics/bronze/raw_data/iot_data_raw.csv` (SOURCE_DATA_PATH)
5. **Compute** will automatically use serverless (no configuration needed)

**Option B: Using UI**
1. **Workflows** → **Jobs** → **Create Job**
2. **Task type**: Python script
3. **Path**: Path to your Python file (e.g., from Git source: `src/data_processing/bronze_ingestion.py`)
4. **Parameters** (in task settings):
   - Parameter 1: `hydroponics` (DATABRICKS_CATALOG)
   - Parameter 2: `/Volumes/hydroponics/bronze/raw_data/iot_data_raw.csv` (SOURCE_DATA_PATH)
5. **Compute**: Select serverless

#### 3. Run the Job

**CLI:**
```bash
databricks jobs run-now --job-id <job-id>
```

**UI:** Workflows → Jobs → Run Now

#### 4. Verify Results

```sql
USE CATALOG your_catalog;
SELECT COUNT(*) FROM bronze.iot_data;
SELECT COUNT(*) FROM silver.iot_data;
SELECT COUNT(*) FROM gold.iot_data;
SELECT COUNT(*) FROM gold.dim_time;
SELECT COUNT(*) FROM gold.dim_equipment;
```

## Configuration

### Job Parameters

Set these as parameters in the job task configuration:
- **Parameter 1**: `DATABRICKS_CATALOG` (e.g., `hydroponics`)
- **Parameter 2**: `SOURCE_DATA_PATH` (e.g., `/Volumes/hydroponics/bronze/raw_data/iot_data_raw.csv`)

These are passed to the Python script via `sys.argv` and can also be set as environment variables as a fallback.

### Data Paths

All data is stored in Unity Catalog Volumes:
- Source data: `/Volumes/{catalog}/bronze/raw_data/iot_data_raw.csv`
- Bronze: `/Volumes/{catalog}/bronze/iot_data`
- Silver: `/Volumes/{catalog}/silver/iot_data`
- Gold: `/Volumes/{catalog}/gold/iot_data`

Tables are registered in Unity Catalog:
- `{catalog}.bronze.iot_data`
- `{catalog}.silver.iot_data`
- `{catalog}.gold.iot_data`
- `{catalog}.gold.dim_time`
- `{catalog}.gold.dim_equipment`

## Usage

### Run Full Pipeline

The pipeline runs automatically when the Databricks job is executed. It processes:
1. **Bronze Layer**: Ingests raw CSV data from Unity Catalog Volumes
2. **Silver Layer**: Cleans and validates data
3. **Gold Layer**: Creates fact and dimension tables

### Run Individual Layers

To run individual layers in Databricks notebooks:

```python
# Bronze only
from data_processing.bronze_ingestion import run_bronze_ingestion
run_bronze_ingestion("/Volumes/your_catalog/bronze/raw_data/iot_data_raw.csv")

# Silver only (requires bronze to exist)
from data_processing.silver_processing import run_silver_processing
run_silver_processing()

# Gold only (requires silver to exist)
from data_processing.gold_processing import run_gold_processing
run_gold_processing()
```

## Data Processing Details

### Bronze Layer
- Ingests raw CSV data from Unity Catalog Volumes
- Preserves all original data
- Adds metadata: `ingestion_timestamp`, `source_file`
- Stores in Delta format for ACID transactions

### Silver Layer
- Type conversions (string to numeric, ON/OFF to boolean)
- Data quality validation:
  - pH range: 0-14
  - TDS: non-negative
  - Temperature: -10°C to 50°C
  - Humidity: 0-100%
- Deduplication based on id and timestamp
- Adds `silver_processed_timestamp`

### Gold Layer
- Creates star schema with fact and dimension tables
- Calculates optimal condition indicators
- Enables efficient analytical queries
- Time dimension for temporal analysis
- Equipment dimension for equipment tracking

## Querying the Data

After processing, you can query the tables in Databricks:

```sql
-- Example: Find readings with suboptimal pH
SELECT 
    t.date,
    t.hour,
    f.ph_level,
    f.is_ph_optimal,
    f.air_temperature
FROM your_catalog.gold.iot_data f
JOIN your_catalog.gold.dim_time t
    ON f.timestamp_key = t.timestamp_key
WHERE f.is_ph_optimal = false
ORDER BY t.date DESC, t.hour DESC;
```

## Troubleshooting

**Job fails: "Catalog not found"**
- Verify `DATABRICKS_CATALOG` is set as the first job parameter
- Ensure you have permissions to access the catalog

**Job fails: "File not found"**
- Verify data exists at `/Volumes/{catalog}/bronze/raw_data/iot_data_raw.csv`
- Check `SOURCE_DATA_PATH` in job config matches the actual path

**Job fails: "Module not found"**
- Ensure the Python file path in job configuration is correct
- Verify all Python files are accessible from the job

**Schema creation errors**
- Check that you have CREATE SCHEMA permissions in the catalog
- Verify the catalog name is correct

**Data not found**
- Verify data exists in Unity Catalog Volumes at `/Volumes/{catalog}/bronze/raw_data/`

**Schema errors**
- Check that the source CSV matches the expected format

**Permission errors**
- Verify your job has write access to the specified catalog/schema
